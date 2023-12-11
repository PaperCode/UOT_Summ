import os
import sys
import gc
import glob

import json
import jsonlines

import random


from time import time
from tqdm import tqdm
from os.path import join

import spacy
from spacy.lang.en import English
from spacy.language import Language

from typing import List
import ftfy


@Language.component('custom_sentence_end')
def set_custom_sentence_end_points(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
    return doc


def handle_special_cases(text):
    text = text.encode("ascii", "ignore").decode()

    text = text.replace("fig .", "fig")
    text = text.replace("et al .", "et al")
    text = text.replace("et  al .", "et al")
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = text.replace(".(", " . (")
    text = text.replace(").", ") . ")
    text = text.replace("u.k .", "u.k.")
    text = text.replace(" eq .", " eq.")
    text = text.replace(" ref .", " ref.")
    # text = text.replace("", "")
    text = text.replace("\"", " ")

    return text


def re_sents_splitting_via_spacy(sents_list, nlp_pipeline, discard_length=25):
    concatenated = ""
    for sent in sents_list:
        concatenated = concatenated + sent + ' '

    concatenated = handle_special_cases(concatenated)

    text_processed = nlp_pipeline(concatenated)

    re_splitted_sents = []

    for sent in text_processed.sents:
        if len(sent.text.strip()) < discard_length:  # discard too short sentence
            continue
        if sent.text.count('\\') > 10:  # discard if it contains too many latex fomula
            continue
        re_splitted_sents.append(sent.text)

    return re_splitted_sents


def sents_splitting_via_spacy_4_billsum(sents_in_str, nlp_pipeline, discard_length=25):
    # concatenated = ""
    # for sent in sents_list:
    #     concatenated = concatenated + sent + ' '

    sents_in_str = sents_in_str.replace(": (1)", " . (1)")
    sents_in_str = sents_in_str.replace(", (", " . (")
    sents_in_str = sents_in_str.replace(", and (", " . and (")
    sents_in_str = handle_special_cases(sents_in_str)

    text_processed = nlp_pipeline(sents_in_str)

    splitted_sents = []

    for sent in text_processed.sents:
        if len(sent.text.strip()) < discard_length:  # discard too short sentence
            continue
        # if sent.text.count('\\') > 10:  # discard if it contains too many latex fomula
        #     continue
        splitted_sents.append(sent.text)

    return splitted_sents


def is_irrelevant_section(section_name):
    # 'none'
    irrelevant_name_list = ['conflict of interest', 'conflicts of interest', 'acknowledgement',
                            'financial support', 'supporting information',
                            'declaration of patient consent', 'competing interest', 'disclosure statement',
                            'supplementary', 'reference', 'figure', 'table',
                            'appendi', 'proof of', 'proofs of',
                            'statement of ethic', 'author contribution']
    for name in irrelevant_name_list:
        if section_name.lower().count(name) > 0:
            return True

    return False


def handle_sections(sections_list, section_names_list, nlp_pipeline):  # sections_list: List[List[str]]
    article_sents_num = 0
    processed_sections = []
    processed_section_names = []

    for idx, sec in enumerate(sections_list):
        if is_irrelevant_section(section_names_list[idx]):
            continue
        sec = re_sents_splitting_via_spacy(sec, nlp_pipeline, discard_length=15)
        if len(sec) == 0:
            continue

        processed_sections.append(sec)
        processed_section_names.append(section_names_list[idx])
        article_sents_num = article_sents_num + len(sec)

    return article_sents_num, processed_sections, processed_section_names


def handle_sections_4_billsum(sections_in_str, nlp_pipeline):
    splitted_sections = sections_in_str.split('<SECTION-HEADER>')

    article_sents_num = 0
    processed_sections = []
    processed_section_names = []

    for idx, sec in enumerate(splitted_sections):
        if len(sec) == 0:
            continue
            
        if len(sec.split('.', 1)) == 2:
            (section_name, section_content) = sec.split('.', 1)
        else:
            section_name = 'NO HEADER PROVIDED'
            section_content = sec

        section_in_sents = sents_splitting_via_spacy_4_billsum(section_content, nlp_pipeline, discard_length=15)
        if len(section_in_sents) == 0:
            continue

        processed_sections.append(section_in_sents)
        processed_section_names.append(section_name)
        article_sents_num = article_sents_num + len(section_in_sents)

    return article_sents_num, processed_sections, processed_section_names


def process_pubmed_arxiv(src_dir, save_path):
    nlp_pipeline = English()  # just the language with no model
    nlp_pipeline.add_pipe("sentencizer")
    nlp_pipeline.add_pipe("custom_sentence_end", before="sentencizer")

    # sentencizer = nlp_pipeline.create_pipe("sentencizer")
    # nlp_pipeline.add_pipe(sentencizer)

    splits = glob.glob(os.path.join(src_dir, "*.txt"))
    for split in tqdm(splits, desc="Loading Splits"):
        split_name = os.path.splitext(os.path.basename(split))[0]
        print("split_name", split_name)
        saved_sub_dir = join(save_path, split_name)
        print("saved_sub_dir", saved_sub_dir)
        if not os.path.exists(saved_sub_dir):
            os.makedirs(saved_sub_dir)

        num_articles_skipped = 0

        with open(split, "r") as articles_info:
            # print("split", split)
            # count the number of lines in the file
            try:
                print(
                    "Counting the number of lines for data integrity and accurate progress bar.",
                    " Press CTRL+C to cancel line counting (not recommended)."
                )
                t0 = time()
                num_articles = sum(1 for line in articles_info)
                # reset pointer to the beginning of the file
                print("done in " + str(time() - t0))
            except KeyboardInterrupt:
                num_articles = None
                print("Skipping line counting...")
            articles_info.seek(0)

            for idx, article_info in enumerate(
                tqdm(articles_info, desc="Loading Articles", total=num_articles)
            ):
                article_info = json.loads(article_info)
                abstract_sents = article_info["abstract_text"]
                sections = article_info["sections"]

                # print("abstract_sents", abstract_sents)
                # print("article_sents", article_sents)

                # remove the <S> and </S> tokens
                abstract_sents = [x[4:-4] for x in abstract_sents]
                processed_abstract_sents = re_sents_splitting_via_spacy(abstract_sents, nlp_pipeline,
                                                                        discard_length=20)
                # train must have at least 2 sentence in the abstract
                if (len(processed_abstract_sents) < 2) and (split_name == "train"):
                    num_articles_skipped += 1
                    continue  # move to next article
                    # print("abstract too short!")
                    # print("abstract_sents", abstract_sents)
                    # print("processed_abstract_sents length", len(processed_abstract_sents))
                    # print("processed_abstract_sents", processed_abstract_sents)
                    # print("##################################")
                elif len(processed_abstract_sents) == 0:
                    num_articles_skipped += 1
                    continue  # move to next article

                if (len(sections) != len(article_info["section_names"])) and (split_name == "train"):
                    num_articles_skipped += 1
                    print("unmatched sections!")
                    print("sections", sections)
                    print("section_names", article_info["section_names"])
                    print("##################################")
                    continue  # move to next article

                article_sents_num, processed_sections, processed_section_names \
                    = handle_sections(sections, article_info["section_names"], nlp_pipeline)
                # assert len(processed_sections) == len(processed_section_names)
                # train must have at least three sentences in the article
                # there are some articles that have one sentence (probably an error during data collection)
                if (article_sents_num <= 3) and (split_name == "train"):
                    num_articles_skipped += 1
                    continue  # move to next article
                    # print("article too short !")
                    # print("article_info", article_info)
                    # print("sections", sections)
                    # print("section_names", article_info["section_names"])
                    # print("article_sents_num", article_sents_num)
                    # print("processed_sections", processed_sections)
                    # print("processed_section_names", processed_section_names)
                    # print("##################################")
                elif article_sents_num == 0:
                    num_articles_skipped += 1
                    continue  # move to next article

                # article sentences must >= abstract sentences
                if (article_sents_num < len(processed_abstract_sents)) and (split_name == "train"):
                    num_articles_skipped += 1
                    # print("articles have less sentences than abstract !")
                    # print("abstract_sents num", len(processed_abstract_sents))
                    # print("article_sents_num", article_sents_num)
                    # print("processed_abstract_sents", processed_abstract_sents)
                    # print("processed_sections", processed_sections)
                    # print("sections", sections)
                    # print("##################################")
                    continue  # move to next article

                json_obj = dict()
                json_obj["id"] = article_info["article_id"]
                json_obj["section_names"] = processed_section_names
                json_obj["abstract"] = processed_abstract_sents
                json_obj["sections"] = processed_sections  # List[List[str]]

                json_4_write = json.dumps(json_obj, indent=4)
                with open(join(saved_sub_dir, article_info["article_id"]+'.json'), 'a') as output_json_file:
                    # print(json_obj["id"])
                    output_json_file.write(json_4_write)

        print(str(num_articles_skipped) + " articles are skipped in " + split_name)


def process_billsum(src_dir, save_path, train_proportion=0.9):
    total_indices = list(range(18949))
    random.seed(20)
    random.shuffle(total_indices)
    train_set_size = int(18949 * train_proportion)
    train_set_indices = total_indices[:train_set_size]
    # print("train_set_indices", train_set_indices)
    # print("train_set_indices", len(train_set_indices))

    nlp_pipeline = English()  # just the language with no model
    nlp_pipeline.add_pipe("sentencizer")
    nlp_pipeline.add_pipe("custom_sentence_end", before="sentencizer")

    data_file_list = ["us_train_data_final_OFFICIAL.jsonl",
                      "us_test_data_final_OFFICIAL.jsonl"]

    for data_file in data_file_list:
        with jsonlines.open(os.path.join(src_dir, data_file)) as reader:
            line_counter = 0
            for one_line in reader:
                # print("one_line", one_line)
                json_obj = dict()
                json_obj["id"] = one_line['bill_id']

                processed_abstract_sents = \
                    sents_splitting_via_spacy_4_billsum(one_line['clean_summary'], nlp_pipeline, discard_length=20)
                if len(processed_abstract_sents) == 0:
                    print("processed_abstract_sents", processed_abstract_sents)
                    print("----------------------------")
                    continue  # move to next article

                article_sents_num, processed_sections, processed_section_names \
                    = handle_sections_4_billsum(one_line['clean_text'], nlp_pipeline)
                if article_sents_num == 0:
                    print("processed_sections", processed_sections)
                    print("----------------------------")
                    continue  # move to next article

                json_obj["abstract"] = processed_abstract_sents
                json_obj["section_names"] = processed_section_names
                json_obj["sections"] = processed_sections

                json_4_write = json.dumps(json_obj, indent=4)

                if data_file == 'us_train_data_final_OFFICIAL.jsonl':
                    if line_counter in train_set_indices:
                        saved_sub_dir = join(save_path, 'train')
                    else:
                        saved_sub_dir = join(save_path, 'val')
                else:
                    saved_sub_dir = join(save_path, 'test')

                if not os.path.exists(saved_sub_dir):
                    os.makedirs(saved_sub_dir)

                with open(join(saved_sub_dir, json_obj["id"] + '.json'), 'a') as output_json_file:
                    # print(json_obj["id"])
                    output_json_file.write(json_4_write)

                # print("line_counter", line_counter)
                line_counter += 1


def handle_paragraphs_4_gov_report(paras_list, nlp_pipeline, para_discard_length=150, sent_discard_length=20):
    concatenated = ""
    for para in paras_list:
        if len(para) > para_discard_length:
            concatenated = concatenated + para + ' '

    concatenated = handle_special_cases(concatenated)

    text_processed = nlp_pipeline(concatenated)

    re_splitted_sents = []

    for sent in text_processed.sents:
        if len(sent.text.strip()) < sent_discard_length:  # discard too short sentence
            continue
        if sent.text.count('\\') > 10:  # discard if it contains too many latex fomula
            continue
        re_splitted_sents.append(sent.text)

    return re_splitted_sents


# **NOTE:** For experiments using GAO reports, we do not include the paragraphs in the Letter section
# (its subsections are included).
def recursive_item_generator_neglect_section(json_input, lookup_key, neglected_section="Letter"):
    neglect_flag = False
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if v == neglected_section:
                neglect_flag = True
                continue
            if neglect_flag:
                neglect_flag = False
                continue
            if k == lookup_key:
                yield v
            else:
                yield from recursive_item_generator_neglect_section(v, lookup_key, neglected_section="Letter")
    elif isinstance(json_input, list):
        for item in json_input:
            yield from recursive_item_generator_neglect_section(item, lookup_key, neglected_section="Letter")


def recursive_item_generator(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                yield v
            else:
                yield from recursive_item_generator(v, lookup_key)
    elif isinstance(json_input, list):
        for item in json_input:
            yield from recursive_item_generator(item, lookup_key)


def process_crs(src_dir, save_path):
    nlp_pipeline = English()  # just the language with no model
    nlp_pipeline.add_pipe("sentencizer")
    nlp_pipeline.add_pipe("custom_sentence_end", before="sentencizer")

    src_dataset_splits = ["train", "valid", "test"]
    tgt_dataset_splits = ["train", "val", "test"]

    for (src_sp, tgt_sp) in zip(src_dataset_splits, tgt_dataset_splits):
        saved_sub_dir = join(save_path, tgt_sp)
        if not os.path.exists(saved_sub_dir):
            os.makedirs(saved_sub_dir)

        num_articles_skipped = 0
        with open(join(src_dir, "split_ids", "crs_" + src_sp + ".ids"), "r") as file_name_list:
            # print(file_name_list)
            for one_file in file_name_list:
                one_file = one_file.rstrip()
                # print(one_file)
                with open(join(src_dir, 'crs', one_file + ".json")) as input_json_file:
                    data = json.load(input_json_file)
                    # print("data", data)

                    tgt_part = data["summary"]

                    processed_abstract_sents = re_sents_splitting_via_spacy(tgt_part, nlp_pipeline, discard_length=20)
                    # print("len(processed_abstract_sents)", len(processed_abstract_sents))
                    if len(processed_abstract_sents) == 0:
                        num_articles_skipped += 1
                        print(one_file)
                        print("summary", data["summary"])
                        continue  # move to next article

                    src_part = data["reports"]
                    paragraph_generator = recursive_item_generator(json_input=src_part, lookup_key="paragraphs")
                    paragraph_list = []
                    for paras in paragraph_generator:
                        # print("paras", paras)
                        paras_in_sents = handle_paragraphs_4_gov_report(paras, nlp_pipeline)
                        if len(paras_in_sents) > 0:
                            paragraph_list.append(paras_in_sents)

                    # print("len(paragraph_list)", len(paragraph_list))

                    if len(paragraph_list) == 0:
                        num_articles_skipped += 1
                        print(one_file)
                        print("reports", data["reports"])
                        continue  # move to next article

                    json_obj = dict()
                    json_obj["id"] = data["id"]
                    json_obj["title"] = data["title"]
                    json_obj["abstract"] = processed_abstract_sents
                    json_obj["sections"] = paragraph_list  # List[List[str]]

                    json_4_write = json.dumps(json_obj, indent=2)
                    with open(join(saved_sub_dir, data["id"] + '.json'), 'a') as output_json_file:
                        # print(json_obj["id"])
                        output_json_file.write(json_4_write)

        print(str(num_articles_skipped) + " articles are skipped in " + src_sp + " of crs.")


def process_gao(src_dir, save_path):
    nlp_pipeline = English()  # just the language with no model
    nlp_pipeline.add_pipe("sentencizer")
    nlp_pipeline.add_pipe("custom_sentence_end", before="sentencizer")

    src_dataset_splits = ["train", "valid", "test"]
    tgt_dataset_splits = ["train", "val", "test"]

    for (src_sp, tgt_sp) in zip(src_dataset_splits, tgt_dataset_splits):
        saved_sub_dir = join(save_path, tgt_sp)
        if not os.path.exists(saved_sub_dir):
            os.makedirs(saved_sub_dir)

        num_articles_skipped = 0
        with open(join(src_dir, "split_ids", "gao_" + src_sp + ".ids"), "r") as file_name_list:
            # print(file_name_list)
            for one_file in file_name_list:
                one_file = one_file.rstrip()
                # print(one_file)
                with open(join(src_dir, 'gao', one_file + ".json")) as input_json_file:
                    data = json.load(input_json_file)
                    # print("data", data)

                    tgt_part = data["highlight"]

                    tgt_paragraph_generator = recursive_item_generator(json_input=tgt_part, lookup_key="paragraphs")
                    tgt_paragraph_list = []
                    for paras in tgt_paragraph_generator:
                        tgt_paragraph_list.extend(paras)
                    processed_abstract_sents = re_sents_splitting_via_spacy(
                        tgt_paragraph_list, nlp_pipeline, discard_length=20)
                    # print("len(processed_abstract_sents)", len(processed_abstract_sents))
                    if len(processed_abstract_sents) == 0:
                        num_articles_skipped += 1
                        print(one_file)
                        print("highlight", data["highlight"])
                        continue  # move to next article

                    src_part = data["report"]
                    paragraph_generator = recursive_item_generator_neglect_section(
                        json_input=src_part, lookup_key="paragraphs", neglected_section="Letter")
                    paragraph_list = []
                    for paras in paragraph_generator:
                        # print("paras", paras)
                        paras_in_sents = handle_paragraphs_4_gov_report(paras, nlp_pipeline)
                        if len(paras_in_sents) > 0:
                            paragraph_list.append(paras_in_sents)
                    # print("len(paragraph_list)", len(paragraph_list))

                    if len(paragraph_list) == 0:
                        num_articles_skipped += 1
                        print(one_file)
                        print("report", data["report"])
                        continue  # move to next article

                    json_obj = dict()
                    json_obj["id"] = data["id"]
                    json_obj["title"] = data["title"]
                    json_obj["abstract"] = processed_abstract_sents
                    json_obj["sections"] = paragraph_list  # List[List[str]]

                    json_4_write = json.dumps(json_obj, indent=2)
                    with open(join(saved_sub_dir, data["id"] + '.json'), 'a') as output_json_file:
                        # print(json_obj["id"])
                        output_json_file.write(json_4_write)

        print(str(num_articles_skipped) + " articles are skipped in " + src_sp + " of gao.")


def sanitize_story_line(line):
    line = ftfy.fix_encoding(line)

    # sentence_endings = [".", "!", "?", "...", "'", "`", '"', ")", "\u2019", "\u201d"]
    sentence_endings = ["!", "?", "'", "`", '"', ")"]

    # Highlight are essentially bullet points and don't have proper sentence endings
    # if line[-1] not in sentence_endings:
    #     line += "."

    if line[-1] != '.':
        if line[-1] in sentence_endings:
            line = line[:-1] + '.'
        else:
            line += "."

    return line


def read_source_sections(article_sections_list):
    # todo handle too short sentence -> no
    # article_sections_list = article_sections_list[0: self._reader_settings['max_src_sections_nums']]
    sections_in_str: List[str] = []

    sent_num = 0
    for one_section in article_sections_list:
        sent_num += len(one_section)
        one_section_in_str = ''
        for one_sent in one_section:
            # one_sent = one_sent.strip("\n").lower()
            # words_one_sent = one_sent.split()
            one_sent = one_sent.strip()

            if len(one_sent) == 0:
                continue
            else:
                one_sent = sanitize_story_line(one_sent)
                one_section_in_str = one_section_in_str + one_sent + ' '

        if len(one_section_in_str) > 0:
            sections_in_str.append(one_section_in_str)

    # print("tokenized_sections", tokenized_sections)
    # print("article_sents_list", article_sents_list)

    token_num = len((" ".join(sections_in_str)).split())

    return token_num, sent_num


def read_abstract_sents(abstract_sents_list):
    # todo handle too short sentence ??

    abstract_sents_in_str = []
    for one_sent in abstract_sents_list:
        # one_sent = one_sent.strip("\n").lower()
        one_sent = one_sent.strip()

        if len(one_sent) == 0:
            continue
        else:
            one_sent = sanitize_story_line(one_sent)

        if len(one_sent) > 0:
            abstract_sents_in_str.append(one_sent)

    return len((" ".join(abstract_sents_in_str)).split())


def dataset_statistics():
    import statistics
    dataset_path_dic = {
        # 'pubmed': '/gds/xshen/projdata11/researchPJ/ot_abs/processed_pubmed/',
        # 'arxiv': '/gds/xshen/projdata11/researchPJ/ot_abs/processed_arxiv/',
        # 'gov': '/gds/xshen/projdata17/researchPJ/processed_gov/',
        'billsum': '/gds/xshen/projdata17/researchPJ/nle_1st_revision/processed_billsum/'
    }
    splits_names = ['train', 'val', 'test']

    for data_name, data_path in dataset_path_dic.items():

        src_token_num_list = []
        src_sent_num_list = []
        src_section_num_list = []

        tgt_token_num_list = []
        tgt_sent_num_list = []

        for split in splits_names:
            file_list = list(os.listdir(os.path.join(data_path, split)))

            for one_input_file in file_list:

                with open(os.path.join(data_path, split, one_input_file), 'r') as input_json_file:
                    data = json.load(input_json_file)
                    src_token_num, src_sent_num = read_source_sections(data['sections'])

                    tgt_token_num = read_abstract_sents(data['abstract'])

                    src_token_num_list.append(src_token_num)
                    src_sent_num_list.append(src_sent_num)
                    src_section_num_list.append(len(data['sections']))

                    tgt_token_num_list.append(tgt_token_num)
                    tgt_sent_num_list.append(len(data['abstract']))

        print("data_name", data_name)

        print("average token number in src ", statistics.mean(src_token_num_list))
        print("average sentence number in src ", statistics.mean(src_sent_num_list))
        print("average section number in src ", statistics.mean(src_section_num_list))

        print("average token number in tgt ", statistics.mean(tgt_token_num_list))
        print("average sentence number in tgt ", statistics.mean(tgt_sent_num_list))
        print('***************')


if __name__ == "__main__":
    task_name = sys.argv[1]  # process or statistics

    if task_name == "process":
        dataset_name = sys.argv[2]
        src_articles_dir = sys.argv[3]

        # path = "arxiv-pubmed"
        # path = "pubmed"

        # The path where the articles are to be saved
        tgt_path = sys.argv[4]
        if not os.path.exists(tgt_path):
            os.makedirs(tgt_path)

        if (dataset_name == "arxiv") or (dataset_name == "pubmed"):
            process_pubmed_arxiv(src_articles_dir, tgt_path)
        elif dataset_name == "gov":
            # process_crs(src_articles_dir, tgt_path)  # process crs
            process_gao(src_articles_dir, tgt_path)  # process gao
        elif dataset_name == "billsum":
            process_billsum(src_articles_dir, tgt_path, train_proportion=0.9)  # process billsum
        else:
            print("wrong dataset_name!")
    elif task_name == "statistics":
        dataset_statistics()
    else:
        print("wrong task name!")

''' 


python preprocess_with_sections.py billsum process ./billsum/clean_final/ ./processed_billsum

/gds/xshen/projdata17/Download/goverment_report/gov-report/
/gds/xshen/projdata17/researchPJ/processed_gov/  

        # process_one_gov_report(src_articles_dir, tgt_path)
        
        def process_one_gov_report(file_path, save_path):
    with open(file_path) as input_json_file:
        data = json.load(input_json_file)
        # print(data)

        result = recursive_item_generator(json_input=data, lookup_key="paragraphs")
        result_list = []
        for res in result:
            result_list.append(res)
            print("res", res)

        json_4_write = json.dumps(data, indent=2)
        with open(join(save_path, "tmp6.json"), 'a') as output_json_file:
            output_json_file.write(json_4_write)

    return 0
'''