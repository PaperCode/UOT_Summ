from pathlib import Path
from typing import Dict, Optional, List
import json

import logging
import os
import glob
import hashlib
import ftfy

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

# new 4 ot


@DatasetReader.register("pubmed_arxiv")
class PubmedArxivDatasetReader(DatasetReader):
    """
    Reads the CNN/DailyMail dataset for text summarization.

    The output of `read` is a list of `Instance` s with the fields:
        source_tokens : `TextField` and
        target_tokens : `TextField`

    # Parameters

    source_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to `SpacyTokenizer()`.
    target_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to `source_tokenizer`.
    source_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input (source side) token representations. Defaults to
        `{"tokens": SingleIdTokenIndexer()}`.
    target_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define output (target side) token representations. Defaults to
        `source_token_indexers`.
    source_max_tokens : `int`, optional
        Maximum number of tokens in source sequence.
    target_max_tokens : `int`, optional
        Maximum number of tokens in target sequence.
    source_prefix : `str`, optional
        An optional prefix to prepend to source strings. For example, with a T5 model,
        you want to set the `source_prefix` to "summarize: ".
    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        # source_max_tokens: Optional[int] = None,
        # target_max_tokens: Optional[int] = None,
        source_prefix: Optional[str] = None,

        reader_settings: Optional[Dict] = None,

        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        # self._source_max_tokens = source_max_tokens
        # self._target_max_tokens = target_max_tokens
        self._source_prefix = source_prefix

        # 4 ot
        self._reader_settings = reader_settings

    @staticmethod
    def _hashhex(url):
        h = hashlib.sha1()
        h.update(url)
        return h.hexdigest()

    @staticmethod
    def _sanitize_story_line(line):
        line = ftfy.fix_encoding(line)

        sentence_endings = [".", "!", "?", "...", "'", "`", '"', ")", "\u2019", "\u201d"]

        # CNN stories always start with "(CNN)"
        # if line.startswith("(CNN)"):
        #     line = line[len("(CNN)") :]

        # Highlight are essentially bullet points and don't have proper sentence endings
        if line[-1] not in sentence_endings:
            line += "."

        return line

    # @staticmethod
    def _read_source_sections(self, article_sections_list):
        # todo handle too short sentence -> no
        article_sections_list = article_sections_list[0: self._reader_settings['max_src_sections_nums']]
        sections_in_str = []
        for one_section in article_sections_list:
            one_section_in_str = ''
            for one_sent in one_section:
                # one_sent = one_sent.strip("\n").lower()
                # words_one_sent = one_sent.split()
                one_sent = one_sent.strip()

                if len(one_sent) == 0:
                    continue
                else:
                    one_sent = PubmedArxivDatasetReader._sanitize_story_line(one_sent)
                    one_section_in_str = one_section_in_str + one_sent + ' '

                # if len(words_one_section) + len(words_one_sent) + 1 <= \
                #         self._reader_settings['max_len_one_section_src']:  # 1 for eos
                #     words_one_section.extend(words_one_sent)
                #     words_one_section.append(cfg.W_EOS)
                # else:
                #     continue

            # words = words[0:cfg.MAX_LEN_ONE_SECTION_SRC]
            # if len(words_one_section) > 0:
            #     tokenized_sections.append(words_one_section)
            if len(one_section_in_str) > 0:
                sections_in_str.append(one_section_in_str)

        # print("tokenized_sections", tokenized_sections)
        # print("article_sents_list", article_sents_list)

        return sections_in_str

    def _read_abstract_sents(self, abstract_sents_list):
        # todo handle too short sentence ??
        abstract_sents_list = abstract_sents_list[0: self._reader_settings['max_abst_sentences_nums']]

        abstract_sents_in_str = []
        for one_sent in abstract_sents_list:
            # one_sent = one_sent.strip("\n").lower()
            one_sent = one_sent.strip()

            if len(one_sent) == 0:
                continue
            else:
                one_sent = PubmedArxivDatasetReader._sanitize_story_line(one_sent)

            if len(one_sent) > 0:
                abstract_sents_in_str.append(one_sent)

            # words = one_sent.split()
            # words = words[0: self._reader_settings['max_len_one_sentence_abst'] - 1]
            # words.append(cfg.W_EOS)
            # tokenized_sents.append(words)

        # print("abstract_sents_list", abstract_sents_list)
        # print("tokenized_sents", tokenized_sents)
        return abstract_sents_in_str

    def _read_one_file(self, one_file_path: str):
        with open(one_file_path) as input_json_file:
            data = json.load(input_json_file)

            source_sections = self._read_source_sections(data['sections'])
            abstract_sents = self._read_abstract_sents(data['abstract'])
            if len(source_sections) > 0 and len(abstract_sents) > 0:
                return source_sections, abstract_sents
            else:
                return None

        # article: List[str] = []
        # summary: List[str] = []
        # highlight = False
        #
        # with open(story_path, "r") as f:
        #     for line in f:
        #         line = line.strip()
        #         if line == "":
        #             continue
        #
        #         if line == "@highlight":
        #             highlight = True
        #             continue
        #         line = PubmedArxivDatasetReader._sanitize_story_line(line)
        #         (summary if highlight else article).append(line)
        #
        # return " ".join(article), " ".join(summary)

    @staticmethod
    def _strip_extension(filename: str) -> str:
        return os.path.splitext(filename)[0]

    @overrides
    def _read(self, file_path: str):
        # print("file_path\n", file_path)
        # Reset exceeded counts
        # self._source_max_exceeded = 0
        # self._target_max_exceeded = 0

        # url_file_path = cached_path(file_path, extract_archive=True)
        # data_dir = os.path.join(os.path.dirname(url_file_path), "..")
        # cnn_stories_path = os.path.join(data_dir, "cnn_stories")
        # dm_stories_path = os.path.join(data_dir, "dm_stories")
        # print("dm_stories_path", dm_stories_path)

        file_set = {Path(s).stem for s in glob.glob(os.path.join(file_path, "*.json"))}
        # cnn_stories = {Path(s).stem for s in glob.glob(os.path.join(cnn_stories_path, "*.story"))}
        # dm_stories = {Path(s).stem for s in glob.glob(os.path.join(dm_stories_path, "*.story"))}
        # print("file_set", file_set)

        # for one_json_file in file_set:
        for file_idx, one_json_file in enumerate(file_set):
            one_json_file_path = os.path.join(file_path, one_json_file) + ".json"
            # print("************************************************************")
            # print("one_json_file_path\n", one_json_file_path)
            retrieved_one_file = self._read_one_file(one_json_file_path)
            if retrieved_one_file is not None:
                article_sections, summary_sents = retrieved_one_file
            else:
                continue

            # if len(article_sections) == 0 or len(summary_sents) == 0 or len(article_sections) < len(summary_sents):
            #     continue
            # print("article_sections\n", article_sections)
            # print("summary_sents\n", summary_sents)

            yield self.text_to_instance(file_idx, article_sections, summary_sents)

        # with open(url_file_path, "r") as url_file:
        #     for url in self.shard_iterable(url_file):
        #         url = url.strip()
        #
        #         url_hash = self._hashhex(url.encode("utf-8"))
        #
        #         if url_hash in cnn_stories:
        #             story_base_path = cnn_stories_path
        #         elif url_hash in dm_stories:
        #             story_base_path = dm_stories_path
        #         else:
        #             raise ConfigurationError(
        #                 "Story with url '%s' and hash '%s' not found" % (url, url_hash)
        #             )
        #
        #         story_path = os.path.join(story_base_path, url_hash) + ".story"
        #         # article, summary = self._read_story(story_path)
        #         article, summary = self._read_one_file(story_path)
        #
        #         if len(article) == 0 or len(summary) == 0 or len(article) < len(summary):
        #             continue
        #
        #         yield self.text_to_instance(article, summary)

    @overrides
    def text_to_instance(
        self,
        file_idx: int,
        source_sequence_list: List[str],
        target_sequence_list: List[str] = None,
        # self, source_sequence: str, target_sequence: str = None
    ) -> Instance:  # type: ignore
        # if self._source_prefix is not None:  # self._source_prefix : none
        #     tokenized_source = self._source_tokenizer.tokenize(
        #         self._source_prefix + source_sequence
        #     )
        # else:
        #     tokenized_source = self._source_tokenizer.tokenize(source_sequence)
        # print("file_idx", file_idx)
        file_id_field = MetadataField({"file_id": file_idx})

        tokenized_source_sections: List[TextField] = []
        for source_sequence in source_sequence_list:
            # print("source_sequence", source_sequence)
            tokenized_source = self._source_tokenizer.tokenize(source_sequence)

            # print("tokenized_source", tokenized_source)
            # if self._source_max_tokens is not None and len(tokenized_source) > self._source_max_tokens:
            #     tokenized_source = tokenized_source[: self._source_max_tokens]
            # self._source_max_tokens: 1022
            if len(tokenized_source) > self._reader_settings['max_len_one_section_src']:
                tokenized_source = tokenized_source[: self._reader_settings['max_len_one_section_src']]
            tokenized_source_sections.append(TextField(tokenized_source))
        source_sections_field = ListField(tokenized_source_sections)

        if target_sequence_list is not None:
            tokenized_target_sents: List[TextField] = []
            for target_sequence in target_sequence_list:
                # print("target_sequence", target_sequence)
                tokenized_target = self._target_tokenizer.tokenize(target_sequence)
                # print("tokenized_target", tokenized_target)

                # self._target_max_tokens： 54
                if len(tokenized_target) > self._reader_settings['max_len_one_sentence_abst']:
                    tokenized_target = tokenized_target[: self._reader_settings['max_len_one_sentence_abst']]
                tokenized_target_sents.append(TextField(tokenized_target))

            target_sents_field = ListField(tokenized_target_sents)
            return Instance({"metadata": file_id_field,
                             "source_sections_field": source_sections_field,
                             "target_sents_field": target_sents_field})
        else:
            return Instance({"metadata": file_id_field,
                             "source_sections_field": source_sections_field})

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        # instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
        # if "target_tokens" in instance.fields:
        #     instance.fields["target_tokens"]._token_indexers = self._target_token_indexers  # type: ignore
        for text_field in instance["source_sections_field"].field_list:
            text_field.token_indexers = self._source_token_indexers
        if "target_sents_field" in instance.fields:
            for text_field in instance["target_sents_field"].field_list:
                text_field.token_indexers = self._target_token_indexers


'''
    @staticmethod
    def _read_story(story_path: str):
        article: List[str] = []
        summary: List[str] = []
        highlight = False

        with open(story_path, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue

                if line == "@highlight":
                    highlight = True
                    continue
                line = CNNDailyMailDatasetReader._sanitize_story_line(line)
                (summary if highlight else article).append(line)

        return " ".join(article), " ".join(summary)
        
'''