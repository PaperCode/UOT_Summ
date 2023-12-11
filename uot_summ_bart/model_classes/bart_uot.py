import warnings
from typing import Dict, Tuple, Any, cast, Optional
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.models.model import Model

# from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

# https://github.com/allenai/allennlp/blob/main/allennlp/nn/beam_search.py
# from allennlp.nn.beam_search import BeamSearch
from .beam_search_4_uot import BeamSearch

from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import ROUGE, BLEU
from allennlp.common.lazy import Lazy

import torch
from torch import nn
import torch.nn.functional as F

from .ot_matcher import OTMatcher
from .utils_4_uot import *
import os
from rouge_score import rouge_scorer

# from transformers.models.bart.modeling_bart import BartModel, BartForConditionalGeneration
from .modeling_bart_4_uot import BartForConditionalGeneration4UOT
# from ..ref_files.ref_modeling_bart import BartForConditionalGeneration

DecoderCacheType = Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...]


@Model.register("bart_uot")
class BartUOT(Model):
    """
    BART model from the paper "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,
    Translation, and Comprehension" (https://arxiv.org/abs/1910.13461). The Bart model here uses a language
    modeling head and thus can be used for text generation.

    # Parameters

    model_name : `str`, required
        Name of the pre-trained BART model to use. Available options can be found in
        `transformers.models.bart.modeling_bart.BART_PRETRAINED_MODEL_ARCHIVE_MAP`.
    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies.
    beam_search : `Lazy[BeamSearch]`, optional (default = `Lazy(BeamSearch)`)
        This is used to during inference to select the tokens of the decoded output sequence.
    indexer : `PretrainedTransformerIndexer`, optional (default = `None`)
        Indexer to be used for converting decoded sequences of ids to to sequences of tokens.
    encoder : `Seq2SeqEncoder`, optional (default = `None`)
        Encoder to used in BART. By default, the original BART encoder is used.
    """

    def __init__(
        self,
        model_name: str,
        vocab: Vocabulary,
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch),
        indexer: PretrainedTransformerIndexer = None,
        # encoder: Seq2SeqEncoder = None,
        model_settings: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(vocab)
        # print("model_name", model_name)
        # self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        # self.bart_4_uot = BartForConditionalGeneration.from_pretrained(model_name)
        self.bart_4_uot = BartForConditionalGeneration4UOT.from_pretrained(model_name)  # todo new 4 uot summ

        self._indexer = indexer or PretrainedTransformerIndexer(model_name, namespace="tokens")

        self._start_id = self.bart_4_uot.config.bos_token_id  # CLS
        self._decoder_start_id = self.bart_4_uot.config.decoder_start_token_id or self._start_id
        self._end_id = self.bart_4_uot.config.eos_token_id  # SEP
        self._pad_id = self.bart_4_uot.config.pad_token_id  # PAD
        print("self._start_id", self._start_id)  # CLS 0
        print("self._decoder_start_id", self._decoder_start_id)  # 2
        print("self._end_id", self._end_id)  # SEP 2
        print("self._pad_id", self._pad_id)  # PAD 1

        print("beam_search", beam_search)
        # At prediction time, we'll use a beam search to find the best target sequence.
        # For backwards compatibility, check if beam_size or max_decoding_steps were passed in as
        # kwargs. If so, update the BeamSearch object before constructing and raise a DeprecationWarning
        deprecation_warning = (
            "The parameter {} has been deprecated."
            " Provide this parameter as argument to beam_search instead."
        )
        beam_search_extras = {}
        if "beam_size" in kwargs:
            beam_search_extras["beam_size"] = kwargs["beam_size"]
            warnings.warn(deprecation_warning.format("beam_size"), DeprecationWarning)
        if "max_decoding_steps" in kwargs:
            beam_search_extras["max_steps"] = kwargs["max_decoding_steps"]
            warnings.warn(deprecation_warning.format("max_decoding_steps"), DeprecationWarning)

        # print("beam_search_extras", beam_search_extras)
        # print("beam_search_extras", **beam_search_extras)
        self._beam_search = beam_search.construct(
            end_index=self._end_id, vocab=self.vocab, **beam_search_extras
        )

        self._rouge = ROUGE(exclude_indices={self._start_id, self._pad_id, self._end_id})
        # self._bleu = BLEU(exclude_indices={self._start_id, self._pad_id, self._end_id})

        # todo new 4 uot summ
        self._model_settings = model_settings
        self._ot_matcher = OTMatcher(input_dimension=model_settings['ot_matcher_input_size'],
                                     dim_section_ext=256, hidden_size_ot_extractor=128)
        self._optimizer_4_ot_matcher = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._ot_matcher.parameters()),
            lr=1e-5)  # new 4 ot

        self._google_rouge_scorer = rouge_scorer.RougeScorer(['rouge3'], use_stemmer=True)  # todo useful

        # Replace bart encoder with given encoder. We need to extract the two embedding layers so that
        # we can use them in the encoder wrapper
        # if encoder is not None:
        #     assert (
        #         encoder.get_input_dim() == encoder.get_output_dim() == self.bart.config.hidden_size
        #     )
        #     self.bart.model.encoder = _BartEncoderWrapper(
        #         encoder,
        #         self.bart.model.encoder.embed_tokens,
        #         self.bart.model.encoder.embed_positions,
        #     )

    @overrides
    def forward(
        self,
        # source_tokens: TextFieldTensors,
        # target_tokens: TextFieldTensors = None,
        metadata: Dict,
        source_sections_field: TextFieldTensors,
        target_sents_field: TextFieldTensors = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward step of Bart.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The source tokens for the encoder. We assume they are stored under the `tokens` key.
        target_tokens : `TextFieldTensors`, optional (default = `None`)
            The target tokens for the decoder. We assume they are stored under the `tokens` key. If no target
            tokens are given, the source tokens are shifted to the right by 1.


        # Returns

        `Dict[str, torch.Tensor]`
            During training, this dictionary contains the `decoder_logits` of shape `(batch_size,
            max_target_length, target_vocab_size)` and the `loss`. During inference, it contains `predictions`
            of shape `(batch_size, max_decoding_steps)` and `log_probabilities` of shape `(batch_size,)`.

        """
        # print("metadata", metadata)

        # inputs = source_tokens
        # targets = target_tokens
        torch.set_printoptions(profile="full")
        # print("source_sections_field", source_sections_field.size())
        # print("source_sections_field", source_sections_field)
        # print("source_sections_field token_ids", source_sections_field["tokens"]["token_ids"].size())
        # print("source_sections_field mask", source_sections_field["tokens"]["mask"].size())
        # print("source_sections_field type_ids", source_sections_field["tokens"]["type_ids"].size())

        # print("target_sents_field", target_sents_field.size())
        # print("target_sents_field", target_sents_field)
        # print("target_sents_field token_ids", target_sents_field["tokens"]["token_ids"].size())
        # print("target_sents_field mask", target_sents_field["tokens"]["mask"].size())
        # print("target_sents_field type_ids", target_sents_field["tokens"]["type_ids"].size())

        inputs = source_sections_field
        targets = target_sents_field

        input_ids, input_mask = inputs["tokens"]["token_ids"], inputs["tokens"]["mask"]

        # print("input_ids", input_ids.size())
        # print("input_mask", input_mask.size())

        # If no targets are provided, then shift input to right by 1. Bart already does this internally
        # but it does not use them for loss calculation.
        if targets is not None:
            target_ids, target_mask = targets["tokens"]["token_ids"], targets["tokens"]["mask"]
        # else:  # not possible for our case!
            # target_ids = input_ids[:, 1:]
            # target_mask = input_mask[:, 1:]

        if self.training:
            outputs = self._forward_4_training(input_ids, input_mask, target_ids, target_mask)
        else:
            outputs = self._forward_4_inference(input_ids, input_mask, target_ids, file_id=metadata[0]["file_id"])

        return outputs

    def _forward_4_training(
        self, input_ids, input_mask, target_ids, target_mask
    ):

        outputs = {}
        for p in self._ot_matcher.parameters():
            p.requires_grad = False

        bart_outputs = self.bart_4_uot(
            input_ids=input_ids,
            attention_mask=input_mask,
            # decoder_input_ids=target_ids[:, :-1].contiguous(),
            decoder_input_ids=target_ids,
            # decoder_attention_mask=target_mask[:, :-1].contiguous(),
            decoder_attention_mask=target_mask,
            use_cache=False,
            return_dict=True,
            uot_matcher=self._ot_matcher,
            recombined_max_len_abst=self._model_settings["recombined_max_len_abst_one_sec"],
        )
        outputs["decoder_logits"] = bart_outputs.logits

        # print("recombined_target_ids QQ", bart_outputs["recombined_target_ids"].size())
        # print("recombined_target_attention_mask QQ", bart_outputs["recombined_target_attention_mask"].size())
        # The BART paper mentions label smoothing of 0.1 for sequence generation tasks
        # todo add squeeze
        outputs["loss"] = sequence_cross_entropy_with_logits(
            bart_outputs.logits,
            # cast(torch.LongTensor, target_ids[:, 1:].contiguous()),
            # cast(torch.BoolTensor, target_mask[:, 1:].contiguous()),
            cast(torch.LongTensor,
                 (bart_outputs["recombined_target_ids"][:, 1:]).contiguous()),
            cast(torch.BoolTensor,
                 (bart_outputs["recombined_target_attention_mask"][:, 1:]).contiguous()),
            label_smoothing=0.1,
            average="token",
        )

        for p in self._ot_matcher.parameters():
            p.requires_grad = True

        self._ot_matcher.zero_grad()
        loss_ot_matcher = self._ot_matcher(encoded_sections=bart_outputs["encoder_last_hidden_state"],
                                           mask_x=input_mask.squeeze(0),
                                           labels=bart_outputs["src_marginal"])

        print("loss_ot_matcher", loss_ot_matcher)

        loss_ot_matcher.backward()
        torch.nn.utils.clip_grad_norm_(self._ot_matcher.parameters(), 2.0)  # TODO ??
        self._optimizer_4_ot_matcher.step()

        return outputs

    def id_tensor_2_list_with_sent_split(self, indices_tensor, short_sent_threshold=6):
        indices_to_remove = [self._start_id, self._pad_id, self._end_id]

        indices_list = indices_tensor.tolist()
        # print("indices_list", indices_list)

        sents_list = []
        for para_list in indices_list:
            one_sent_list = []
            for token_index in para_list:
                if token_index in indices_to_remove:
                    if len(one_sent_list) > short_sent_threshold:
                        sents_list.append(one_sent_list)
                    one_sent_list = []
                else:
                    one_sent_list.append(token_index)
            if len(one_sent_list) > short_sent_threshold:
                sents_list.append(one_sent_list)

        return sents_list

    # indices_tensor: 1d
    def remove_special_indices(self, indices_tensor):
        indices_to_remove = [self._start_id, self._pad_id, self._end_id]
        for ind2remove in indices_to_remove:
            preserved = torch.nonzero(1 - (indices_tensor == ind2remove).float(), as_tuple=True)[0]
            indices_tensor = torch.index_select(indices_tensor, dim=0, index=preserved)

        return indices_tensor

    def _forward_4_inference(
        self,
        input_ids, input_mask, target_ids, file_id,
    ):
        input_ids = input_ids.squeeze(0)
        input_mask = input_mask.squeeze(0)
        target_ids = target_ids.squeeze(0)
        # print("file_id", file_id)

        # print("input_ids", input_ids.size())
        # print("input_ids", input_ids)
        # print("input_mask", input_mask.size())
        # print("input_mask", input_mask)
        # print("target_ids", target_ids.size())
        # print("target_ids", target_ids)

        encoded_sections = self.bart_4_uot.model.encoder(
            input_ids=input_ids,
            attention_mask=input_mask,
        )
        encoded_sections = encoded_sections["last_hidden_state"]
        # print("encoded_sections yy", encoded_sections.size())

        scaling_factors = self._ot_matcher.predict_alignment_score_4_sections(
            emb_x=encoded_sections, mask_x=input_mask)
        # print("scaling_factors", scaling_factors)
        predicted_summ_sents_nums = torch.floor(scaling_factors + 0.5)  # todo !!!!!!!!!!!!!!!!!!! tune constant here
        # predicted_summ_sents_nums = torch.floor(scaling_factors + 0.7)
        # print("predicted_summ_sents_nums", predicted_summ_sents_nums)

        # todo for beam_search
        outputs = {}

        # Use decoder start id and start of sentence to start decoder
        initial_decoder_ids = torch.tensor(
            [[self._decoder_start_id]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).repeat(input_ids.shape[0], 1)

        # print("initial_decoder_ids", initial_decoder_ids.size())
        # print("initial_decoder_ids", initial_decoder_ids)

        inital_state = {
            "input_ids": input_ids,
            "input_mask": input_mask,
        }

        beam_result = self._beam_search.search(
            initial_decoder_ids, inital_state, self.take_step,
            summ_sents_nums_to_generate=predicted_summ_sents_nums,
        )
        # print("beam_result", beam_result)
        # print("beam_result[0].size()", beam_result[0].size())
        # print("beam_result[1].size()", beam_result[1].size())
        # print("beam_result[1]", beam_result[1])

        predictions = beam_result[0]
        max_pred_indices = (
            beam_result[1].argmax(dim=-1).view(-1, 1, 1).expand(-1, -1, predictions.shape[-1])
        )
        predictions = predictions.gather(dim=1, index=max_pred_indices).squeeze(dim=1)
        # print("max_pred_indices", max_pred_indices)
        # print("predictions", predictions.size())
        # print("predictions", predictions)

        # select_indices = torch.nonzero((predicted_summ_sents_nums > 0.5).float(), as_tuple=True)[0]
        # print("select_indices", select_indices)
        # predictions_positive = torch.index_select(predictions, dim=0, index=select_indices)  # todo consider
        # print("predictions_positive", predictions_positive.size())

        predictions_cat = self.remove_special_indices(torch.flatten(predictions))
        target_ids_cat = self.remove_special_indices(torch.flatten(target_ids))
        # print("predictions_cat", predictions_cat)
        # print("target_ids_cat", target_ids_cat)

        # todo: this version of ROUGE is problematic,
        #   pls use PERL ROUGE to evaluate the dumped files in make_output_human_readable, as the final results.
        # allennlp rouge https://github.com/allenai/allennlp/issues/5153
        self._rouge(predictions_cat.unsqueeze(0), target_ids_cat.unsqueeze(0))
        # self._rouge(predictions, target_ids)
        # self._bleu(predictions, target_ids)

        outputs["predictions"] = predictions

        # todo useless??
        # outputs["log_probabilities"] = (
        #     beam_result[1].gather(dim=-1, index=max_pred_indices[..., 0]).squeeze(dim=-1)
        # )

        self.make_output_human_readable(outputs, target_ids, file_id)
        return outputs

    # print("layer_cache[0].size()", layer_cache[0].size())
    # print("layer_cache[1].size()", layer_cache[1].size())
    # print("layer_cache", layer_cache)
    @staticmethod
    def _decoder_cache_to_dict(decoder_cache: DecoderCacheType) -> Dict[str, torch.Tensor]:
        cache_dict = {}
        for layer_index, layer_cache in enumerate(decoder_cache):
            # Each layer caches the key and value tensors for its self-attention and cross-attention.
            # Hence the `layer_cache` tuple has 4 elements.
            assert len(layer_cache) == 4
            for tensor_index, tensor in enumerate(layer_cache):
                key = f"decoder_cache_{layer_index}_{tensor_index}"
                cache_dict[key] = tensor
        return cache_dict

    def _dict_to_decoder_cache(self, cache_dict: Dict[str, torch.Tensor]) -> DecoderCacheType:
        decoder_cache = []
        for layer_index in range(len(self.bart_4_uot.model.decoder.layers)):
            base_key = f"decoder_cache_{layer_index}_"
            layer_cache = (
                cache_dict[base_key + "0"],
                cache_dict[base_key + "1"],
                cache_dict[base_key + "2"],
                cache_dict[base_key + "3"],
            )
            decoder_cache.append(layer_cache)
        assert decoder_cache
        return tuple(decoder_cache)

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        # Parameters

        last_predictions : `torch.Tensor`
            The predicted token ids from the previous step. Shape: `(group_size,)`
        state : `Dict[str, torch.Tensor]`
            State required to generate next set of predictions
        step : `int`
            The time step in beam search decoding.


        # Returns

        `Tuple[torch.Tensor, Dict[str, torch.Tensor]]`
            A tuple containing logits for the next tokens of shape `(group_size, target_vocab_size)` and
            an updated state dictionary.
        """
        # print("last_predictions.shape", last_predictions.shape)

        if len(last_predictions.shape) == 1:
            last_predictions = last_predictions.unsqueeze(-1)
        # print("take_step state", state)

        decoder_cache = None
        decoder_cache_dict = {
            k: state[k].contiguous()
            for k in state
            if k not in {"input_ids", "input_mask", "encoder_states"}
        }
        # print("decoder_cache_dict", decoder_cache_dict)

        if len(decoder_cache_dict) != 0:
            decoder_cache = self._dict_to_decoder_cache(decoder_cache_dict)
            # print("decoder_cache uu", decoder_cache)

        encoder_outputs = (state["encoder_states"],) if "encoder_states" in state else None
        # print("encoder_outputs df", encoder_outputs)

        outputs = self.bart_4_uot(  # todo need change
            input_ids=state["input_ids"] if encoder_outputs is None else None,
            attention_mask=state["input_mask"],
            encoder_outputs=encoder_outputs,
            decoder_input_ids=last_predictions,
            past_key_values=decoder_cache,
            use_cache=True,
            return_dict=True,
        )
        # print("outputs pf", outputs)

        logits = outputs.logits[:, -1, :]
        log_probabilities = F.log_softmax(logits, dim=-1)

        decoder_cache = outputs.past_key_values
        if decoder_cache is not None:
            decoder_cache_dict = self._decoder_cache_to_dict(decoder_cache)
            state.update(decoder_cache_dict)

        state["encoder_states"] = outputs.encoder_last_hidden_state

        # print("***********************************")
        return log_probabilities, state

    @overrides
    def make_output_human_readable(
        self,
        output_dict: Dict[str, torch.Tensor],
        target_ids: torch.Tensor,
        file_id: int,
        path_4_generation: str =
        "/gds/xshen/projdata17/researchPJ/bart_summ/result4rouge/reconstructed_12epoch_epsilon0.05_tau0.3_gov_v2_const0.5_no3gramBolck/",
        # path_4_generation: str = "/gds/xshen/projdata17/researchPJ/bart_summ/result4rouge/gov_v1_const0.5/",
    ) -> Dict[str, Any]:
        """

        # Parameters

        output_dict : `Dict[str, torch.Tensor]`
            A dictionary containing a batch of predictions with key `predictions`. The tensor should have
            shape `(batch_size, max_sequence_length)`

        # Returns

        `Dict[str, Any]`
            Original `output_dict` with an additional `predicted_tokens` key that maps to a list of lists of
            tokens.

        """
        if not os.path.exists(path_4_generation):
            os.makedirs(path_4_generation)
        ground_truth_path = os.path.join(path_4_generation, 'ground_truth')
        summary_path = os.path.join(path_4_generation, 'summary')
        if not os.path.exists(ground_truth_path):
            os.makedirs(ground_truth_path)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)

        predictions = output_dict["predictions"]
        predicted_tokens = [None] * predictions.shape[0]
        for i in range(predictions.shape[0]):
            predicted_tokens[i] = self._indexer.indices_to_tokens(
                {"token_ids": predictions[i].tolist()},
                self.vocab,
            )
        output_dict["predicted_tokens"] = predicted_tokens  # type: ignore
        # print("predicted_tokens", output_dict["predicted_tokens"])

        predictions_in_list = self.id_tensor_2_list_with_sent_split(predictions, short_sent_threshold=6)
        target_ids_in_list = self.id_tensor_2_list_with_sent_split(target_ids, short_sent_threshold=1)
        # print("predictions_in_list", predictions_in_list)
        # print("target_ids_in_list", target_ids_in_list)

        # output_dict["predicted_text"] = self._indexer._tokenizer.batch_decode(
        #     predictions.tolist(), skip_special_tokens=True
        # )
        # https://huggingface.co/transformers/internal/tokenization_utils.html
        predicted_text = self._indexer._tokenizer.batch_decode(
            predictions_in_list, skip_special_tokens=True
        )
        # print("predicted_text", predicted_text)
        # todo add trigram blocking
        # predicted_text_trigram_blocked = \
        #     remove_repetitive_sents(predicted_text, self._google_rouge_scorer,
        #                             rouge_name='rouge3', remove_threshold=0.6)
        # print("predicted_text_trigram_blocked", predicted_text_trigram_blocked)

        # output_dict["predicted_text"] = predicted_text_trigram_blocked
        output_dict["predicted_text"] = predicted_text

        ground_truth_text = self._indexer._tokenizer.batch_decode(
            target_ids_in_list, skip_special_tokens=True
        )

        # print("predicted_text", output_dict["predicted_text"])
        # print("ground_truth_text", ground_truth_text)

        with open("".join((summary_path, '/', str(file_id))), "w") as f:
            for idx, sent in enumerate(output_dict["predicted_text"]):
                sent = sent.strip()
                f.write(sent) if idx == len(output_dict["predicted_text"]) - 1 else f.write(sent + "\n")

        with open("".join((ground_truth_path, '/', str(file_id))), "w") as f:
            for idx, sent in enumerate(ground_truth_text):
                sent = sent.strip()
                f.write(sent) if idx == len(ground_truth_text) - 1 else f.write(sent + "\n")

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not self.training:
            metrics.update(self._rouge.get_metric(reset=reset))
            # metrics.update(self._bleu.get_metric(reset=reset))
        return metrics

    default_predictor = "seq2seq"


# useless ?
'''
@Seq2SeqEncoder.register("bart_encoder")
class BartEncoder(Seq2SeqEncoder):
    """
    The BART encoder, implemented as a `Seq2SeqEncoder`, which assumes it operates on
    already embedded inputs.  This means that we remove the token and position embeddings
    from BART in this module.  For the typical use case of using BART to encode inputs to your
    model (where we include the token and position embeddings from BART), you should use
    `PretrainedTransformerEmbedder(bart_model_name, sub_module="encoder")` instead of this.

    # Parameters

    model_name : `str`, required
        Name of the pre-trained BART model to use. Available options can be found in
        `transformers.models.bart.modeling_bart.BART_PRETRAINED_MODEL_ARCHIVE_MAP`.
    """

    def __init__(self, model_name):
        super().__init__()

        bart = BartModel.from_pretrained(model_name)
        self.hidden_dim = bart.config.hidden_size
        self.bart_encoder = bart.encoder
        self.bart_encoder.embed_tokens = lambda x: x
        self.bart_encoder.embed_positions = lambda x: torch.zeros(
            (x.shape[0], x.shape[1], self.hidden_dim), dtype=torch.float32
        )

    @overrides
    def get_input_dim(self) -> int:
        return self.hidden_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor):
        # The first element is always the last encoder states for each input token.
        # Depending on the config, the second output will contain a list of the encoder states
        # after each transformer layer. Similarly, the third output can contain the attentions from each layer.
        # We only care about the first element.
        return self.bart_encoder(input_ids=inputs, attention_mask=mask)[0]


class _BartEncoderWrapper(nn.Module):
    """
    A wrapper class for a `Seq2SeqEncoder` allowing it to replace the encoder in `Bart`.
    This class is only used internally by `Bart`.
    """

    def __init__(
        self, encoder: Seq2SeqEncoder, embed_tokens: nn.Embedding, embed_positions: nn.Embedding
    ):
        """
        # Parameters

        encoder : `Seq2SeqEncoder`, required
            Encoder to be used by `Bart`.
        embed_tokens : `nn.Embedding`, required
            The token embedding layer of the BART model.
        embed_positions : `nn.Embedding`, required
            The positional embedding layer of the BART model.

        """
        super().__init__()
        self.encoder = encoder
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        # AllenNLP Seq2SeqEncoder's don't necessarily return those and the encoder might not even use
        # Attention, thus ensure those are not expected.
        # assert not bart_config.output_attentions
        # assert not bart_config.output_hidden_states

    @overrides
    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        x = self.embed_tokens(input_ids) + self.embed_positions(input_ids)
        encoder_states = self.encoder(x, attention_mask)
        # The last two elements are attention and history of hidden states, respectively
        return encoder_states, [], []
'''

