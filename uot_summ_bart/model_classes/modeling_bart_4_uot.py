from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
    BartPretrainedModel,
    BartEncoder,
    BartDecoder,
    shift_tokens_right,
)
# BartModel,

from transformers.models.bart.configuration_bart import BartConfig
# Seq2SeqLMOutput,
from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
    BaseModelOutput,
)


# from transformers.file_utils import (
#     add_code_sample_docstrings,
#     add_end_docstrings,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     replace_return_docstrings,
# )

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import cast

from .utils_4_uot import *
from allennlp.nn.util import sequence_cross_entropy_with_logits


# @add_start_docstrings(
#     "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
# )
# class BartForConditionalGeneration4UOT(BartForConditionalGeneration):
'''
        **bos_token_id** (:obj:`int`, `optional`)) -- The id of the `beginning-of-stream` token.
        - **forced_bos_token_id** (:obj:`int`, `optional`) -- The id of the token to force as the first generated token
          after the :obj:`decoder_start_token_id`. Useful for multilingual models like :doc:`mBART
          <../model_doc/mbart>` where the first generated token needs to be the target language token.
'''


class BartForConditionalGeneration4UOT(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)

        # self.model = BartModel(config)
        self.model = BartModel4UOT(config)  # new

        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self._eos_token_id = config.eos_token_id  # SEP 2
        self._pad_id = config.pad_token_id  # PAD 1
        # self._start_id = config.bos_token_id  # CLS 0
        self._start_id = config.bos_token_id  # CLS
        # self._decoder_start_id = config.decoder_start_token_id or self._start_id
        # assert config.bos_token_id == 0
        # self._end_id = self.bart_4_uot.config.eos_token_id  # SEP

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # todo: ??? check whether [CLS], [UNK], [SEP] are the same across different models.
    # todo: true: [CLS], [SEP] false: [PAD]
    def filter_and_recombine_aligned_section_summary_pairs(
            self, aligned_section_summary_indices,
            attention_mask, decoder_input_ids, encoder_last_hidden_state, decoder_attention_mask,
            x_sections_num, recombined_max_len_abst):
        # print("aligned_section_summary_indices", aligned_section_summary_indices)
        # print("attention_mask", attention_mask.size())
        # print("attention_mask", attention_mask)
        # print("encoder_last_hidden_state", encoder_last_hidden_state.size())
        # print("encoder_last_hidden_state", encoder_last_hidden_state)
        # print("decoder_input_ids", decoder_input_ids.size())
        # print("decoder_input_ids", decoder_input_ids)
        # print("decoder_attention_mask", decoder_attention_mask.size())
        # print("decoder_attention_mask", decoder_attention_mask)

        # print("recombined_max_len_abst", recombined_max_len_abst)
        # print("x_sections_num", x_sections_num)

        selection_mask_4_x, selection_mask_4_y = zip(*aligned_section_summary_indices)
        # print("selection_mask_4_x aa", selection_mask_4_x)
        # print("selection_mask_4_y aa", selection_mask_4_y)
        num_pairs_one_doc = len(list(selection_mask_4_x))
        selection_mask_4_x = torch.LongTensor(list(selection_mask_4_x)).to(encoder_last_hidden_state.device)
        selection_mask_4_y = list(selection_mask_4_y)

        # print("selection_mask_4_x bb", selection_mask_4_x)
        # print("selection_mask_4_y bb", selection_mask_4_y)

        selected_attention_mask = torch.index_select(attention_mask, dim=0, index=selection_mask_4_x)
        selected_encoder_outputs = torch.index_select(encoder_last_hidden_state, dim=0, index=selection_mask_4_x)
        # print("selected_attention_mask", selected_attention_mask.size())
        # print("selected_encoder_outputs", selected_encoder_outputs.size())

        # print("self.device", self.device)
        recombined_decoder_input_ids = (torch.ones(num_pairs_one_doc, recombined_max_len_abst, dtype=torch.long) *
                                        self._pad_id).to(encoder_last_hidden_state.device)
        recombined_decoder_attention_mask = torch.zeros(num_pairs_one_doc, recombined_max_len_abst, dtype=torch.bool).\
            to(encoder_last_hidden_state.device)
        # print("recombined_decoder_input_ids ss", recombined_decoder_input_ids.size())
        # print("recombined_decoder_attention_mask ss", recombined_decoder_attention_mask.size())

        summ_sents_len_one_doc = (torch.sum(decoder_attention_mask.long(), dim=1, keepdim=False)).tolist()
        # print("summ_sents_len_one_doc", summ_sents_len_one_doc)

        recombined_y_len_one_doc = []
        for sec_idx, summ_idx_4_sec in enumerate(selection_mask_4_y):
            # print("sec_idx", sec_idx)
            # print("summ_idx_4_sec", summ_idx_4_sec)
            accumulated_len_y = 0
            for summ_sen_idx in summ_idx_4_sec:
                # print("summ_sen_idx", summ_sen_idx)
                if accumulated_len_y + summ_sents_len_one_doc[summ_sen_idx] <= recombined_max_len_abst:
                    if accumulated_len_y == 0:
                        recombined_decoder_input_ids[sec_idx, 0] = self._start_id
                        recombined_decoder_attention_mask[sec_idx, 0] = True
                        accumulated_len_y = 1

                    recombined_decoder_input_ids[sec_idx, accumulated_len_y: (accumulated_len_y +
                                                 summ_sents_len_one_doc[summ_sen_idx] - 1)] = \
                        decoder_input_ids[summ_sen_idx, 1:summ_sents_len_one_doc[summ_sen_idx]]
                    recombined_decoder_attention_mask[sec_idx, accumulated_len_y: (accumulated_len_y +
                                                      summ_sents_len_one_doc[summ_sen_idx] - 1)] = \
                        decoder_attention_mask[summ_sen_idx, 1:summ_sents_len_one_doc[summ_sen_idx]]
                    accumulated_len_y = accumulated_len_y + summ_sents_len_one_doc[summ_sen_idx] - 1

                    if decoder_input_ids[summ_sen_idx, summ_sents_len_one_doc[summ_sen_idx] - 1] != self._eos_token_id:
                        recombined_decoder_input_ids[sec_idx, accumulated_len_y] = self._eos_token_id
                        recombined_decoder_attention_mask[sec_idx, accumulated_len_y] = True
                        accumulated_len_y = accumulated_len_y + 1

                    # print("accumulated_len_y", accumulated_len_y)
            recombined_y_len_one_doc.append(accumulated_len_y)

        # print("recombined_y_len_one_doc", recombined_y_len_one_doc)

        max_accumulated_len_y = max(recombined_y_len_one_doc)
        # print("max_accumulated_len_y", max_accumulated_len_y)
        recombined_decoder_input_ids = recombined_decoder_input_ids[:, 0:max_accumulated_len_y]
        recombined_decoder_attention_mask = recombined_decoder_attention_mask[:, 0:max_accumulated_len_y]

        # print("recombined_decoder_input_ids", recombined_decoder_input_ids.size())
        # print("recombined_decoder_input_ids", recombined_decoder_input_ids)
        # print("recombined_decoder_attention_mask", recombined_decoder_attention_mask.size())
        # print("recombined_decoder_attention_mask", recombined_decoder_attention_mask)

        return selected_encoder_outputs, selected_attention_mask, \
            recombined_decoder_input_ids, recombined_decoder_attention_mask

    @staticmethod
    @torch.no_grad()
    def get_section_summary_alignment_index(alignment_plan, x_sections_num, y_sents_nums):
        # print("x_sections_num", x_sections_num)
        # print("y_sents_nums", y_sents_nums)
        # print("alignment_plan", alignment_plan.size())
        # print("alignment_plan", alignment_plan)

        aligned_sections = torch.argmax(alignment_plan, dim=0)

        # print("aligned_sections", aligned_sections.size())
        # print("aligned_sections", aligned_sections)

        section_summary_indices = []
        # aligned_sections = aligned_sections_index[doc_idx, :y_sents_nums[doc_idx]]
        # print("aligned_sections", aligned_sections)
        for sec_idx in range(x_sections_num):
            aligned_summ_sent_indices = \
                torch.nonzero((aligned_sections == sec_idx).float(), as_tuple=True)[0].tolist()
            if len(aligned_summ_sent_indices) > 0:
                section_summary_indices.append((sec_idx, aligned_summ_sent_indices))
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        # print("section_summary_indices", section_summary_indices)
        # (section index, (summ sentence indices))
        return section_summary_indices

    @torch.no_grad()
    def find_alignment_with_uot(
            self, source_mask, target_mask, encoder_hidden_state, target_input_ids,
            original_decoder_input_ids, original_decoder_attention_mask,
            decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds,
            use_cache, output_attentions, return_dict, output_hidden_states,
            uot_matcher,
    ):
        # print("source_mask", source_mask.size())
        # print("target_mask", target_mask.size())
        # print("encoder_hidden_state UU", encoder_hidden_state.size())
        # print("target_input_ids", target_input_ids.size())
        # print("source_mask", source_mask)
        # print("target_mask", target_mask)
        # print("encoder_hidden_state", encoder_hidden_state)
        # print("target_input_ids", target_input_ids)

        # x_sections_num = source_mask.float().sum(dim=1)
        # y_sents_nums = target_mask.float().sum(dim=1)
        x_sections_num = source_mask.size(0)
        y_sents_nums = target_mask.size(0)
        # print("x_sections_num", x_sections_num)
        # print("y_sents_nums", y_sents_nums)

        cost_matrix_list = []
        for section_idx in range(x_sections_num):
            encoder_hidden_state_one_sec_repeated = \
                (encoder_hidden_state[section_idx, :, :]).repeat(y_sents_nums, 1, 1)
            # print("encoder_hidden_state_one_sec_repeated", encoder_hidden_state_one_sec_repeated.size())
            source_mask_one_sec_repeated = (source_mask[section_idx, :]).repeat(y_sents_nums, 1)
            # print("source_mask_one_sec_repeated", source_mask_one_sec_repeated.size())

            decoder_outputs_one_sec = self.model.decoder(  # todo ???
                input_ids=target_input_ids,
                attention_mask=target_mask,
                encoder_hidden_states=encoder_hidden_state_one_sec_repeated,
                encoder_attention_mask=source_mask_one_sec_repeated,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # print("decoder_outputs_one_sec", decoder_outputs_one_sec["last_hidden_state"].size())
            # print("decoder_outputs_one_sec", decoder_outputs_one_sec)

            lm_logits = self.lm_head(decoder_outputs_one_sec["last_hidden_state"]) + self.final_logits_bias
            # print("lm_logits PP", lm_logits.size())

            # masked_lm_loss = None
            # if labels is not None:
            #     loss_fct = CrossEntropyLoss()
            #     masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # if not return_dict:
            #     output = (lm_logits,) + outputs[1:]
            #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            # print("original_decoder_input_ids", original_decoder_input_ids.size())
            # print("original_decoder_attention_mask", original_decoder_attention_mask.size())

            # sequence_cross_entropy_with_logits_pseudo_batched sequence_cross_entropy_with_logits
            # note: already masked internally
            loss_one_sec = sequence_cross_entropy_with_logits(
                lm_logits,
                cast(torch.LongTensor, original_decoder_input_ids[:, 1:].contiguous()),
                cast(torch.BoolTensor, original_decoder_attention_mask[:, 1:].contiguous()),
                label_smoothing=0.1,
                average=None,  # "token"
            )
            # print("loss_one_sec", loss_one_sec)
            # cost_matrix_list += [loss_one_sec]
            cost_matrix_list.append(loss_one_sec)

        # print("cost_matrix_list", cost_matrix_list)
        cost_matrix = torch.stack(cost_matrix_list, dim=0)
        cost_matrix = cost_matrix.clone().detach()

        cost_matrix = cost_matrix / torch.max(cost_matrix)  # for numerical stability

        # print("cost_matrix", cost_matrix)
        # todo add del 1

        scaling_factors = uot_matcher.predict_alignment_score_4_sections(
            emb_x=encoder_hidden_state, mask_x=source_mask)
        # print("scaling_factors", scaling_factors)

        # encoded_sections_duplicate = encoded_sections.clone().detach()  # todo: ??

        alignment_plan, _, src_marginal = compute_sinkhorn_loss(
            cost_matrix.unsqueeze(0), scaling_factors.unsqueeze(0),
            torch.ones(1, y_sents_nums).to(scaling_factors.get_device()),
            epsilon=0.006, tau=0.03) 
        alignment_plan = alignment_plan.squeeze(0)
        src_marginal = src_marginal.squeeze(0)
        # print("alignment_plan", alignment_plan.size())
        # print("alignment_plan", alignment_plan)
        # print("src_marginal", src_marginal)

        section_summary_indices = self.get_section_summary_alignment_index(
            alignment_plan.clone().detach(), x_sections_num, y_sents_nums)
        # print("section_summary_indices", section_summary_indices)

        return section_summary_indices, src_marginal

    # @overrides
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        uot_matcher=None,
        recombined_max_len_abst=200,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        # torch.set_printoptions(profile="full")
        if self.training:  # new 4 ot
            original_decoder_input_ids = (decoder_input_ids.clone().detach()).squeeze(0)
            original_decoder_attention_mask = (decoder_attention_mask.clone().detach()).squeeze(0)

            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            decoder_attention_mask = decoder_attention_mask.squeeze(0)

        # print("labels", labels)

        # if labels is not None:  # is None for uot
        #     if decoder_input_ids is None and decoder_inputs_embeds is None:
        #         decoder_input_ids = shift_tokens_right(
        #             labels, self.config.pad_token_id, self.config.decoder_start_token_id
        #         )

        # print("decoder_input_ids 1", decoder_input_ids.size())
        # print("decoder_input_ids 1", decoder_input_ids)
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None:  # todo not suitable for uot
            if decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        else:  # new 4 ot
            if self.training:
                decoder_input_ids = decoder_input_ids.squeeze(0)

        if self.training:  # new 4 ot
            decoder_input_ids = decoder_input_ids[:, :-1].contiguous()
            decoder_attention_mask = decoder_attention_mask[:, :-1].contiguous()

        # print("decoder_input_ids 2", decoder_input_ids.size())
        # print("decoder_input_ids 2", decoder_input_ids)

        # print("output_attentions pp", output_attentions)
        # print("self.config.output_attentions", self.config.output_attentions)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # print("output_attentions ss", output_attentions)

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # print("use_cache 1", use_cache)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # print("use_cache 2", use_cache)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("return_dict XX", return_dict)

        encode_step_results = self.model.wrapped_encode_step(
            input_ids,
            attention_mask=attention_mask,
            # decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            # decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            # decoder_head_mask=decoder_head_mask,
            # cross_attn_head_mask=cross_attn_head_mask,
            # past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            # decoder_inputs_embeds=decoder_inputs_embeds,
            # use_cache=use_cache,             # todo
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print("encode_step_results 99", encode_step_results)
        if self.training:
            aligned_section_summary_indices, src_marginal = self.find_alignment_with_uot(
                source_mask=attention_mask, target_mask=decoder_attention_mask,
                encoder_hidden_state=encode_step_results["last_hidden_state"], target_input_ids=decoder_input_ids,
                original_decoder_input_ids=original_decoder_input_ids,
                original_decoder_attention_mask=original_decoder_attention_mask,
                decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values, decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache, output_attentions=output_attentions, return_dict=return_dict,
                output_hidden_states=output_hidden_states,
                uot_matcher=uot_matcher,
            )
            src_marginal = src_marginal.clone().detach()

            recombined_encoder_outputs, recombined_attention_mask, \
                recombined_decoder_input_ids, recombined_decoder_attention_mask = \
                self.filter_and_recombine_aligned_section_summary_pairs(
                    aligned_section_summary_indices, attention_mask, original_decoder_input_ids,
                    encode_step_results["last_hidden_state"], original_decoder_attention_mask,
                    x_sections_num=attention_mask.size(0), recombined_max_len_abst=recombined_max_len_abst,
                )

            # todo jian cha dao zhe li
            outputs = self.model(
                input_ids=None,  # not used
                attention_mask=recombined_attention_mask,
                decoder_input_ids=(recombined_decoder_input_ids[:, :-1]).contiguous(),
                encoder_outputs=recombined_encoder_outputs,  # todo  encode_step_results
                decoder_attention_mask=(recombined_decoder_attention_mask[:, :-1]).contiguous(),
                head_mask=None,  # not used
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=None,  # not used
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:  # todo infer
            # print("encoder_outputs cvb", encoder_outputs)
            outputs = self.model(
                input_ids=None,  # not used
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encode_step_results["last_hidden_state"],
                decoder_attention_mask=decoder_attention_mask,
                head_mask=None,  # not used
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=None,  # not used
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            src_marginal = None
            recombined_decoder_input_ids = None
            recombined_decoder_attention_mask = None

        # print("outputs MM", outputs)
        # print("outputs.past_key_values", len(outputs.past_key_values))
        # print("outputs.past_key_values[0]", len(outputs.past_key_values[0]))
        # print("outputs.past_key_values", outputs.past_key_values)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # todo check masking
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # print("lm_logits XX", lm_logits.size())

        masked_lm_loss = None
        # if labels is not None:  # is None for uot
        #     loss_fct = CrossEntropyLoss()
        #     masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutputWithRecombinedTargets(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            # todo encoder_last_hidden_state when training: encoder_last_hidden_state for all sections.
            # encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_last_hidden_state=encode_step_results["last_hidden_state"].clone().detach(),
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            recombined_target_ids=recombined_decoder_input_ids,
            recombined_target_attention_mask=recombined_decoder_attention_mask,
            src_marginal=src_marginal,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class BartModel4UOT(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def wrapped_encode_step(
        self,
        input_ids=None,
        attention_mask=None,
        # decoder_input_ids=None,
        # decoder_attention_mask=None,
        head_mask=None,
        # decoder_head_mask=None,
        # cross_attn_head_mask=None,
        encoder_outputs=None,
        # past_key_values=None,
        inputs_embeds=None,
        # decoder_inputs_embeds=None,
        # use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # new 4 ot
        # print("encoder_outputs YY", encoder_outputs)
        # print("input_ids", input_ids.size())
        # print("input_ids", input_ids)
        # print("attention_mask", attention_mask)
        # print("head_mask", head_mask)
        # print("inputs_embeds", inputs_embeds)
        # print("output_attentions", output_attentions)
        # print("output_hidden_states", output_hidden_states)
        # print("return_dict", return_dict)
        # print("attention_mask", attention_mask.size())

        if encoder_outputs is None:  # todo: !!!!!!!!!!!!!!! depends on train or infer
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # print("encoder_outputs yy", encoder_outputs)
            # print("encoder_outputs", encoder_outputs["last_hidden_state"].size())

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            # print("flag dsds")
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            # print("encoder_outputs dsds", encoder_outputs)

        return encoder_outputs

    '''
    output_attentions (:obj:`bool`, `optional`):
    Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
    returned tensors for more detail.
            return_dict (:obj:`bool`, `optional`):
        Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        
    decoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
        Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:
        - 1 indicates the head is **not masked**,
        - 0 indicates the head is **masked**.
    use_cache (:obj:`bool`, `optional`):
        If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
        decoding (see :obj:`past_key_values`).
    '''
    def forward(  # todo no need to change
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )

        # print("return_dict", return_dict)

        # x_sections_num, y_sents_nums,
        # mask_x_list, x_ext_list, hs_all, h0_all,
        # y_list, y_ext_list, y_mask_list, max_ext_len

        # print("decoder_input_ids", decoder_input_ids.size())
        # print("decoder_input_ids", decoder_input_ids)
        # print("decoder_attention_mask", decoder_attention_mask.size())
        # print("decoder_attention_mask", decoder_attention_mask)
        # print("encoder_outputs XX", encoder_outputs.size())
        # print("encoder_outputs XX", encoder_outputs)
        # print("attention_mask", attention_mask.size())
        # print("attention_mask", attention_mask)

        # print("decoder_head_mask", decoder_head_mask)
        # print("cross_attn_head_mask", cross_attn_head_mask)
        # print("past_key_values 1", past_key_values)
        # print("decoder_inputs_embeds", decoder_inputs_embeds)
        # print("use_cache", use_cache)
        # print("output_attentions dc", output_attentions)
        # print("output_hidden_states", output_hidden_states)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(  # todo no need to change
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            # encoder_hidden_states=encoder_outputs[0],
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print("past_key_values 2", past_key_values)
        # print("decoder_outputs.past_key_values 2", decoder_outputs.past_key_values)

        if not return_dict:  # todo ?
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_last_hidden_state=encoder_outputs,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_hidden_states=None,
            # encoder_attentions=encoder_outputs.attentions,
            encoder_attentions=None,
        )


'''
    # @classmethod
    # def from_pretrained(cls, model_name):
    #     return BartForConditionalGeneration.from_pretrained(model_name)

    self.model.tentative_decode(
        source_mask=source_mask, target_mask=target_mask,
        encoder_hidden_state=encoder_hidden_state, target_input_ids=target_input_ids)
                    # todo
                output_attentions=True,

'''