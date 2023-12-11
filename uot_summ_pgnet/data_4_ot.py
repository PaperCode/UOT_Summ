# -*- coding: utf-8 -*-
# pylint: skip-file
import sys
import os
import os.path
import time
from operator import itemgetter
import numpy as np
import pickle
from random import shuffle


class BatchData4OT:
    def __init__(self, flist, modules, consts, options, is_train=True):
        self.file_names = []

        self.batch_size = len(flist)

        self.x_one_batch = []
        self.x_ext_one_batch = []
        self.x_mask_one_batch = []

        self.y_one_batch = []
        self.y_ext_one_batch = []
        self.y_mask_one_batch = []

        self.len_x_one_batch = []
        self.len_y_one_batch = []

        self.original_summarys_one_batch = []

        self.x_ext_words = []

        self.max_ext_len = 0
        ###########################################
        self.x_sections_num = []
        self.y_sents_num = []

        ###########################################

        w2i = modules["w2i"]
        i2w = modules["i2w"]
        dict_size = len(w2i)

        if is_train:
            max_len_x = 0
            max_len_y = 0

        for idx_doc in range(len(flist)):
            if len(flist[idx_doc]) == 3:
                content_tokenized, summaries, fn = flist[idx_doc]  # todo
            else:
                print("ERROR!")
                return

            self.file_names.append(fn)
            # print("contents", contents)
            # print("summaries", summaries)

            summary_tokenized, summary_in_sentence = summaries
            self.original_summarys_one_batch.append(summary_in_sentence)

            self.x_sections_num.append(len(content_tokenized))
            self.y_sents_num.append(len(summary_tokenized))

            x = np.zeros((consts["max_len_one_section_src"], len(content_tokenized)), dtype=np.int64)
            x_ext = np.zeros((consts["max_len_one_section_src"], len(content_tokenized)), dtype=np.int64)
            x_mask = np.zeros((consts["max_len_one_section_src"], len(content_tokenized), 1), dtype=np.int64)

            len_x = []

            xi_oovs = []  # shared by one doc

            for idx_section in range(len(content_tokenized)):
                one_section = content_tokenized[idx_section]
                for idx_word in range(len(one_section)):
                    # some sentences in duc is longer than len_x
                    if idx_word == consts["max_len_one_section_src"]:  # MAX_LEN_ONE_SEN_SRC
                        break
                    w = one_section[idx_word]

                    if w not in w2i:  # OOV
                        if w not in xi_oovs:
                            xi_oovs.append(w)
                        # self.x_ext[idx_word, idx_doc] = dict_size + xi_oovs.index(w)  # 50005, 51000
                        x_ext[idx_word, idx_section] = dict_size + xi_oovs.index(w)  # 50005, 51000

                        w = i2w[modules["lfw_emb"]]
                    else:
                        x_ext[idx_word, idx_section] = w2i[w]

                    x[idx_word, idx_section] = w2i[w]
                    x_mask[idx_word, idx_section, 0] = 1
                # self.len_x.append(np.sum(self.x_mask[:, idx_doc, :]))
                # if np.sum(x_mask[:, idx_section, :]) == 0:
                #     print("content_tokenized", content_tokenized)
                len_x.append(np.sum(x_mask[:, idx_section, :]))

            self.x_ext_words.append(xi_oovs)
            if self.max_ext_len < len(xi_oovs):
                self.max_ext_len = len(xi_oovs)

            len_y = []
            if options["has_y"]:
                y = np.zeros((consts["max_len_one_sentence_abst"], len(summary_tokenized)), dtype=np.int64)
                y_ext = np.zeros((consts["max_len_one_sentence_abst"], len(summary_tokenized)), dtype=np.int64)
                y_mask = np.zeros((consts["max_len_one_sentence_abst"], len(summary_tokenized), 1), dtype=np.int64)

                for idx_sent in range(len(summary_tokenized)):
                    one_sent = summary_tokenized[idx_sent]
                    for idx_word in range(len(one_sent)):
                        w = one_sent[idx_word]

                        if w not in w2i:
                            if w in xi_oovs:
                                y_ext[idx_word, idx_sent] = dict_size + xi_oovs.index(w)
                            else:
                                y_ext[idx_word, idx_sent] = w2i[i2w[modules["lfw_emb"]]]  # unk
                            w = i2w[modules["lfw_emb"]]
                        else:
                            y_ext[idx_word, idx_sent] = w2i[w]
                        y[idx_word, idx_sent] = w2i[w]
                        if not options["is_predicting"]:
                            y_mask[idx_word, idx_sent, 0] = 1
                    len_y.append(len(summary_tokenized[idx_sent]))  # todo??
            else:
                y = y_ext = y_mask = None

            if is_train:
                max_len_x = max(int(np.max(len_x)), max_len_x)
                max_len_y = max(int(np.max(len_y)), max_len_y)

            self.x_one_batch.append(x)
            self.x_ext_one_batch.append(x_ext)
            self.x_mask_one_batch.append(x_mask)

            self.y_one_batch.append(y)
            self.y_ext_one_batch.append(y_ext)
            self.y_mask_one_batch.append(y_mask)

            self.len_x_one_batch.append(len_x)
            self.len_y_one_batch.append(len_y)

        for idx_doc in range(len(flist)):
            if not is_train:  # test
                max_len_x = int(np.max(self.len_x_one_batch[idx_doc]))
                max_len_y = int(np.max(self.len_y_one_batch[idx_doc]))

            self.x_one_batch[idx_doc] = self.x_one_batch[idx_doc][0:max_len_x, :]
            self.x_ext_one_batch[idx_doc] = self.x_ext_one_batch[idx_doc][0:max_len_x, :]  # different from above
            self.x_mask_one_batch[idx_doc] = self.x_mask_one_batch[idx_doc][0:max_len_x, :, :]

            self.y_one_batch[idx_doc] = self.y_one_batch[idx_doc][0:max_len_y, :]
            self.y_ext_one_batch[idx_doc] = self.y_ext_one_batch[idx_doc][0:max_len_y, :]
            self.y_mask_one_batch[idx_doc] = self.y_mask_one_batch[idx_doc][0:max_len_y, :, :]

        # print("self.x_one_batch", self.x_one_batch)
        # print("self.x_ext_one_batch", self.x_ext_one_batch)
        # print("self.x_mask_one_batch", self.x_mask_one_batch)
        # print("self.y_one_batch", self.y_one_batch)
        # print("self.y_ext_one_batch", self.y_ext_one_batch)
        # print("self.y_mask_one_batch", self.y_mask_one_batch)
        #
        # print("self.len_x_one_batch", self.len_x_one_batch)
        # print("self.len_y_one_batch", self.len_y_one_batch)
        #
        # print("self.original_summarys_one_batch", self.original_summarys_one_batch)
        # print("self.x_ext_words", self.x_ext_words)
        #
        # print("self.max_ext_len", self.max_ext_len)
        #
        # print("self.x_sections_num", self.x_sections_num)
        # print("self.y_sents_num", self.y_sents_num)
        # print("=======================")


def get_data_4_ot(xyfn_list, modules, consts, options, is_train=True):
    return BatchData4OT(xyfn_list, modules, consts, options, is_train=is_train)


def batched(x_size, options, consts):
    batch_size = consts["testing_batch_size"] if options["is_predicting"] else consts["batch_size"]
    if options["is_debugging"]:
        x_size = 13
    ids = [i for i in range(x_size)]
    if not options["is_predicting"]:
        shuffle(ids)
    batch_list = []
    batch_ids = []
    for i in range(x_size):
        idx = ids[i]
        batch_ids.append(idx)
        if len(batch_ids) == batch_size or i == (x_size - 1):
            batch_list.append(batch_ids)
            batch_ids = []
    return batch_list, len(ids), len(batch_list)


'''
# batch_size = consts["testing_batch_size"] if options["is_predicting"] else consts["batch_size"]
def batched(x_size, options, consts, epoch_num=0):
    if options["is_predicting"]:
        batch_size = consts["testing_batch_size"]
    else:
        if epoch_num < consts["first_stage_epoch_nums"]:
            batch_size = consts["batch_size_first_stage"]
        else:
            batch_size = consts["batch_size_second_stage"]

    if options["is_debugging"]:
        x_size = 13
    ids = [i for i in range(x_size)]
    if not options["is_predicting"]:
        shuffle(ids)
    batch_list = []
    batch_ids = []
    for i in range(x_size):
        idx = ids[i]
        batch_ids.append(idx)
        if len(batch_ids) == batch_size or i == (x_size - 1):
            batch_list.append(batch_ids)
            batch_ids = []
    return batch_list, len(ids), len(batch_list)
    
        consts["max_len_one_sen_src"]
        consts["max_len_one_sen_abst"]
        consts["max_src_sen_nums"]
        consts["max_abst_sen_nums"]

        # consts["len_x"]-> MAX_LEN_X, 400
        # self.x = np.zeros((consts["len_x"], self.batch_size), dtype=np.int64)
        # self.x_ext = np.zeros((consts["len_x"], self.batch_size), dtype=np.int64)
        # self.x_mask = np.zeros((consts["len_x"], self.batch_size, 1), dtype=np.int64)

        # consts["len_y"]-> MAX_LEN_Y, 100
        # self.y = np.zeros((consts["len_y"], self.batch_size), dtype=np.int64)
        # self.y_ext = np.zeros((consts["len_y"], self.batch_size), dtype=np.int64)
        # self.y_mask = np.zeros((consts["len_y"], self.batch_size, 1), dtype=np.int64)
'''
