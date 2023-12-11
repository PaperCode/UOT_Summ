# -*- coding: utf-8 -*-
import os

import sys
import time
import numpy as np
import pickle
import copy
import random
from random import shuffle
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

# import data as datar
import data_4_ot as data_ot
from rouge_score import rouge_scorer

from model4ot import *
from ot_matcher import *

from utils_pg import *
from configs import *

###########################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, default="train", help="train, continue, tune, or test")
# parser.add_argument("-o", "--oracle", type=str, default="none",
#                     help="none, none_mem_opt, word_oracle, sentence_oracle, or external_oracle")
parser.add_argument("-p", "--pattern", type=str, default="ot",
                    help="none, none_mem_opt, ot")
parser.add_argument("-c", "--cuda_id", type=int, default=0, help="Your intended cuda id")
# parser.add_argument("-n", "--noise", action='store_true', default=False, help="Whether to add Gumbel noise")
parser.add_argument("-e", "--exist_model", type=str, default="cnndm.s2s.gru.gpu0.epoch20.1", \
                    help="your best model name")

args = parser.parse_args()
###########################################

cudaid = args.cuda_id
os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid)

cfg = DeepmindConfigs()
TRAINING_DATASET_CLS = DeepmindTraining
TESTING_DATASET_CLS = DeepmindTesting


def print_basic_info(modules, consts, options):
    if options["is_debugging"]:
        print("\nWARNING: IN DEBUGGING MODE\n")
    if options["copy"]:
        print("USE COPY MECHANISM")
    if options["coverage"]:
        print("USE COVERAGE MECHANISM")
    if options["avg_nll"]:
        print("USE AVG NLL as LOSS")
    else:
        print("USE NLL as LOSS")
    if options["has_learnable_w2v"]:
        print("USE LEARNABLE W2V EMBEDDING")
    if options["is_bidirectional"]:
        print("USE BI-DIRECTIONAL RNN")
    if options["omit_eos"]:
        print("<eos> IS OMITTED IN TESTING DATA")
    if options["prediction_bytes_limitation"]:
        print("MAXIMUM BYTES IN PREDICTION IS LIMITED")
    print("RNN TYPE: " + options["cell"])
    for k in consts:
        print(k + ":", consts[k])


def init_modules():
    init_seeds()

    options = {}

    options["is_debugging"] = False
    ###########################################
    # When options["is_predicting"] = True, true means use validation set for tuning, false is real testing.
    if args.mode == 'train':
        print("Mode: training")
        options["is_predicting"] = False
        options["model_selection"] = False
        options["is_continue"] = False
    elif args.mode == 'continue':
        print("Mode: continue previous train")
        options["is_predicting"] = False
        options["model_selection"] = False
        options["is_continue"] = True
    elif args.mode == 'tune':
        print("Mode: tuning")
        options["is_predicting"] = True
        options["model_selection"] = True
        options["is_continue"] = False
    elif args.mode == 'test':
        print("Mode: testing")
        options["is_predicting"] = True
        options["model_selection"] = False
        options["is_continue"] = False
    else:  # default
        print("ERROR: unknown mode, run training mode by default.")
        options["is_predicting"] = False
        options["model_selection"] = False
        options["is_continue"] = False

    if args.pattern in ["none", "none_mem_opt", "ot"]:
        options["pattern"] = args.pattern
    else:  # default
        options["pattern"] = "ot"

    ###########################################

    options["cuda"] = cfg.CUDA and torch.cuda.is_available()
    options["device"] = torch.device("cuda" if options["cuda"] else "cpu")

    # in config.py
    options["cell"] = cfg.CELL
    options["copy"] = cfg.COPY
    options["coverage"] = cfg.COVERAGE
    options["is_bidirectional"] = cfg.BI_RNN
    options["avg_nll"] = cfg.AVG_NLL

    options["beam_decoding"] = cfg.BEAM_SEARCH  # False for greedy decoding

    assert TRAINING_DATASET_CLS.IS_UNICODE == TESTING_DATASET_CLS.IS_UNICODE
    options["is_unicode"] = TRAINING_DATASET_CLS.IS_UNICODE  # True Chinese dataet
    options["has_y"] = TRAINING_DATASET_CLS.HAS_Y

    options["has_learnable_w2v"] = True
    # omit <eos> and continuously decode until length of sentence reaches MAX_LEN_PREDICT (for DUC testing data)
    options["omit_eos"] = False
    options["prediction_bytes_limitation"] = False if TESTING_DATASET_CLS.MAX_BYTE_PREDICT == None else True

    assert options["is_unicode"] == False

    consts = {}
    #################### 4 OT #######################
    options["is_bidirectional_sec_ext"] = cfg.BI_RNN_SECTION_EXT

    consts["hidden_size_ot_extractor"] = cfg.HIDDEN_SIZE_OT_EXTRACTOR
    consts["dim_section_ext"] = cfg.DIM_SECTION_EXT

    consts["max_len_one_section_src"] = cfg.MAX_LEN_ONE_SECTION_SRC  # plus 1 for eos ??
    consts["max_len_one_sentence_abst"] = cfg.MAX_LEN_ONE_SENTENCE_ABST   # plus 1 for eos ??
    consts["recombined_max_len_abst"] = cfg.RECOMBINED_MAX_LEN_ABST   # plus 1 for eos ??
    consts["max_src_sections_nums"] = cfg.MAX_SRC_SECTIONS_NUMS
    consts["max_abst_sentences_nums"] = cfg.MAX_ABST_SENTENCES_NUMS

    # consts["change_mode_epoch_num"] = cfg.CHANGE_MODE_EPOCH_NUM

    consts["lr_4_matcher"] = cfg.LR4MATCHER

    consts["epsilon"] = cfg.EPSILON
    consts["tau_sinkhorn"] = cfg.TAU_SINKHORN

    # consts["tau_nn"] = cfg.TAU_NN

    consts["bucket_size_4_average"] = cfg.BUCKET_SIZE_4_AVERAGE

    # consts["kesi"] = cfg.KESI

    consts["min_len_predict_one_sec"] = TESTING_DATASET_CLS.MIN_LEN_PREDICT_ONE_SEC
    consts["max_len_predict_one_sec"] = TESTING_DATASET_CLS.MAX_LEN_PREDICT_ONE_SEC
    # consts["max_len_predict_total"] = TESTING_DATASET_CLS.MAX_LEN_PREDICT_TOTAL

    #################### 4 OT #######################

    consts["idx_gpu"] = cudaid

    consts["norm_clip"] = cfg.NORM_CLIP
    consts["dim_x"] = cfg.DIM_X
    consts["dim_y"] = cfg.DIM_Y
    consts["len_x"] = cfg.MAX_LEN_X + 1  # plus 1 for eos
    consts["len_y"] = cfg.MAX_LEN_Y + 1
    consts["num_x"] = cfg.MAX_NUM_X
    consts["num_y"] = cfg.NUM_Y
    consts["hidden_size"] = cfg.HIDDEN_SIZE

    consts["batch_size"] = 5 if options["is_debugging"] else TRAINING_DATASET_CLS.BATCH_SIZE
    # 4 warm start
    # consts["first_stage_epoch_nums"] = TRAINING_DATASET_CLS.FIRST_STAGE_EPOCH_NUMS
    # consts["batch_size_first_stage"] = TRAINING_DATASET_CLS.BATCH_SIZE_FIRST_STAGE
    # consts["batch_size_second_stage"] = TRAINING_DATASET_CLS.BATCH_SIZE_SECOND_STAGE
    # 4 warm start

    if options["is_debugging"]:
        consts["testing_batch_size"] = 1 if options["beam_decoding"] else 2
    else:
        # consts["testing_batch_size"] = 1 if options["beam_decoding"] else TESTING_DATASET_CLS.BATCH_SIZE
        consts["testing_batch_size"] = TESTING_DATASET_CLS.BATCH_SIZE

    # consts["min_len_predict"] = TESTING_DATASET_CLS.MIN_LEN_PREDICT
    # consts["max_len_predict"] = TESTING_DATASET_CLS.MAX_LEN_PREDICT
    consts["max_byte_predict"] = TESTING_DATASET_CLS.MAX_BYTE_PREDICT
    consts["testing_print_size"] = TESTING_DATASET_CLS.PRINT_SIZE

    consts["lr"] = cfg.LR
    consts["beam_size"] = cfg.BEAM_SIZE

    consts["max_epoch"] = 50 if options["is_debugging"] else 50
    consts["print_time"] = 5
    consts["save_epoch"] = 1

    assert consts["dim_x"] == consts["dim_y"]
    assert consts["beam_size"] >= 1

    modules = {}

    # returned = (open(cfg.cc.TRAINING_DATA_PATH + "dic.pkl", "rb"))
    # print("returned", len(returned))

    [_, dic, hfw, w2i, i2w, w2w] = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "dic.pkl", "rb"))
    consts["dict_size"] = len(dic)
    modules["dic"] = dic
    modules["w2i"] = w2i
    modules["i2w"] = i2w
    modules["lfw_emb"] = modules["w2i"][cfg.W_UNK]
    modules["eos_emb"] = modules["w2i"][cfg.W_EOS]
    consts["pad_token_idx"] = modules["w2i"][cfg.W_PAD]

    return modules, consts, options


def beam_decode(fname, batch, model, modules, consts, options, prescribed_decoded_sents_num=1):

    fname = str(fname)

    beam_size = consts["beam_size"]

    decoded_sents_num = [0] * beam_size  # new 4 ot
    # print("decoded_sents_num", decoded_sents_num)

    num_live = 1
    num_dead = 0
    samples = []
    sample_scores = np.zeros(beam_size)

    last_traces = [[]]
    last_scores = torch.FloatTensor(np.zeros(1)).to(options["device"])
    last_states = []

    if options["copy"]:
        x, word_emb, dec_state, x_mask, max_ext_len, oovs = batch
        # x, word_emb, dec_state, x_mask, y, len_y, ref_sents, max_ext_len, oovs = batch
    else:
        x, word_emb, dec_state, x_mask = batch
        # x, word_emb, dec_state, x_mask, y, len_y, ref_sents = batch

    next_y = torch.LongTensor(-np.ones((1, num_live), dtype="int64")).to(options["device"])
    x = x.unsqueeze(1)
    word_emb = word_emb.unsqueeze(1)
    x_mask = x_mask.unsqueeze(1)
    dec_state = dec_state.unsqueeze(0)
    if options["cell"] == "lstm":
        dec_state = (dec_state, dec_state)

    if options["coverage"]:
        acc_att = Variable(torch.zeros(T.transpose(x, 0, 1).size())).to(options["device"])  # B *len(x)
        last_acc_att = []

        # for step in range(consts["max_len_predict"]):
    for step in range(consts["max_len_predict_one_sec"]):
        tile_word_emb = word_emb.repeat(1, num_live, 1)
        tile_x_mask = x_mask.repeat(1, num_live, 1)
        if options["copy"]:
            tile_x = x.repeat(1, num_live)

        if options["copy"] and options["coverage"]:
            y_pred, dec_state, acc_att = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask, tile_x,
                                                           max_ext_len, acc_att)
        elif options["copy"]:
            y_pred, dec_state = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask, tile_x, max_ext_len)
        elif options["coverage"]:
            y_pred, dec_state, acc_att = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask,
                                                           acc_att=acc_att)
        else:
            y_pred, dec_state = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask)
        dict_size = y_pred.shape[-1]
        y_pred = y_pred.view(num_live, dict_size)
        if options["coverage"]:
            acc_att = acc_att.view(num_live, acc_att.shape[-1])

        if options["cell"] == "lstm":
            dec_state = (
            dec_state[0].view(num_live, dec_state[0].shape[-1]), dec_state[1].view(num_live, dec_state[1].shape[-1]))
        else:
            dec_state = dec_state.view(num_live, dec_state.shape[-1])

        cand_scores = last_scores + torch.log(y_pred)  # larger is better
        cand_scores = cand_scores.flatten()
        idx_top_joint_scores = torch.topk(cand_scores, beam_size - num_dead)[1]

        # print("cand_scores", cand_scores.size())
        # print("idx_top_joint_scores", idx_top_joint_scores)

        # print("idx_top_joint_scores", idx_top_joint_scores)
        # print("dict_size", dict_size)
        idx_last_traces = torch.floor_divide(idx_top_joint_scores, dict_size)  # todo changed
        # idx_last_traces = idx_top_joint_scores / dict_size
        idx_word_now = idx_top_joint_scores % dict_size
        top_joint_scores = cand_scores[idx_top_joint_scores]

        traces_now = []
        scores_now = np.zeros((beam_size - num_dead))
        states_now = []
        if options["coverage"]:
            acc_att_now = []
            last_acc_att = []

        for i, [j, k] in enumerate(zip(idx_last_traces, idx_word_now)):
            traces_now.append(last_traces[j] + [k])
            scores_now[i] = copy.copy(top_joint_scores[i])
            if options["cell"] == "lstm":
                states_now.append((copy.copy(dec_state[0][j, :]), copy.copy(dec_state[1][j, :])))
            else:
                states_now.append(copy.copy(dec_state[j, :]))
            if options["coverage"]:
                acc_att_now.append(copy.copy(acc_att[j, :]))

        num_live = 0
        last_traces = []
        last_scores = []
        last_states = []

        # print("len(traces_now)", len(traces_now))
        # print("traces_now", traces_now)
        # print("eos_emb", modules["eos_emb"])
        # print("len(traces_now)", len(traces_now[0]))

        for i in range(len(traces_now)):
            # if traces_now[i][-1] == modules["eos_emb"] and len(traces_now[i]) >= consts["min_len_predict_one_sec"]:
            if (traces_now[i][-1] == modules["eos_emb"]) and \
                    (traces_now[i].count(modules["eos_emb"]) == prescribed_decoded_sents_num):
                # samples.append([str(e.item()) for e in traces_now[i][:-1]])
                samples.append([str(e.item()) for e in traces_now[i]])
                sample_scores[num_dead] = scores_now[i]
                num_dead += 1
            else:
                last_traces.append(traces_now[i])
                last_scores.append(scores_now[i])
                last_states.append(states_now[i])
                if options["coverage"]:
                    last_acc_att.append(acc_att_now[i])
                    num_live += 1
            # print("num_dead", num_dead)
            # print("num_live", num_live)

        if num_live == 0 or num_dead >= beam_size:
            break

        last_scores = torch.FloatTensor(np.array(last_scores).reshape((num_live, 1))).to(options["device"])
        next_y = []
        for e in last_traces:
            eid = e[-1].item()
            if eid in modules["i2w"]:
                next_y.append(eid)
            else:
                next_y.append(modules["lfw_emb"])  # unk for copy mechanism
        # print("next_y", next_y)

        next_y = np.array(next_y).reshape((1, num_live))
        next_y = torch.LongTensor(next_y).to(options["device"])
        if options["cell"] == "lstm":
            h_states = []
            c_states = []
            for state in last_states:
                h_states.append(state[0])
                c_states.append(state[1])
            dec_state = (torch.stack(h_states).view((num_live, h_states[0].shape[-1])), \
                         torch.stack(c_states).view((num_live, c_states[0].shape[-1])))
        else:
            dec_state = torch.stack(last_states).view((num_live, dec_state.shape[-1]))
        if options["coverage"]:
            acc_att = torch.stack(last_acc_att).view((num_live, acc_att.shape[-1]))

        assert num_live + num_dead == beam_size

    if num_live > 0:  # exceed max_len_predict_one_sec
        # print("num_live is positive")
        for i in range(num_live):
            samples.append([str(e.item()) for e in last_traces[i]])
            sample_scores[num_dead] = last_scores[i]
            num_dead += 1

    # weight by length
    for i in range(len(sample_scores)):
        sent_len = float(len(samples[i]))
        sample_scores[i] = sample_scores[i] / sent_len  # avg is better than sum.   #*  math.exp(-sent_len / 10)

    idx_sorted_scores = np.argsort(sample_scores)  # ascending order
    # print("sample_scores", sample_scores)
    # print("idx_sorted_scores", idx_sorted_scores)
    # print("samples", samples)

    # move to the outside
    # if options["has_y"]:
    #     ly = len_y[0]
    #     y_true = y[0 : ly].tolist()
    #     y_true = [str(i) for i in y_true[:-1]]  # delete <eos>

    sorted_samples = []
    sorted_scores = []
    filter_idx = []
    for e in idx_sorted_scores:
        # if len(samples[e]) >= consts["min_len_predict"]:
        if len(samples[e]) >= consts["min_len_predict_one_sec"]:
            filter_idx.append(e)
    if len(filter_idx) == 0:
        filter_idx = idx_sorted_scores
    for e in filter_idx:
        sorted_samples.append(samples[e])
        sorted_scores.append(sample_scores[e])

    # print("sorted_samples", sorted_samples)
    # print("sorted_scores", sorted_scores)
    # print("filter_idx", filter_idx)

    # num_samples = len(sorted_samples)
    # if len(sorted_samples) == 1:
    #     sorted_samples = sorted_samples[0]
    #     num_samples = 1

    # for task with bytes-length limitation
    # todo false in our case
    # if options["prediction_bytes_limitation"]:
    #     for i in range(len(sorted_samples)):
    #         sample = sorted_samples[i]
    #         b = 0
    #         for j in range(len(sample)):
    #             e = int(sample[j])
    #             if e in modules["i2w"]:
    #                 word = modules["i2w"][e]
    #             else:
    #                 word = oovs[e - len(modules["i2w"])]
    #             if j == 0:
    #                 b += len(word)
    #             else:
    #                 b += len(word) + 1
    #             if b > consts["max_byte_predict"]:
    #                 sorted_samples[i] = sorted_samples[i][0 : j]
    #                 break

    decoded_sents = []
    dec_one_sent = []

    for e in sorted_samples[-1]:  # todo omit eos
        e = int(e)
        # print("e", e)
        if e == modules["eos_emb"]:
            decoded_sents.append(dec_one_sent)
            dec_one_sent = []
        else:
            if e in modules["i2w"]:  # if not copy, the word are all in dict
                dec_one_sent.append(modules["i2w"][e])
            else:
                dec_one_sent.append(oovs[e - len(modules["i2w"])])

    # print("decoded_sents", decoded_sents)

    # move to the outside ??
    # write_for_rouge(fname, ref_sents, dec_words, cfg)

    # beam search history for checking
    if not options["copy"]:
        oovs = None

    # BEAM_SUMM_PATH: beam_summary
    # write_summ("".join((cfg.cc.BEAM_SUMM_PATH, fname)), sorted_samples, num_samples,
    #            options, modules["i2w"], oovs, sorted_scores)

    # move to the outside
    # BEAM_GT_PATH: beam_ground_truth
    # write_summ("".join((cfg.cc.BEAM_GT_PATH, fname)), y_true, 1, options, modules["i2w"], oovs)

    # print("***********beam end************")

    # decoded_one_sent_in_str = ' '.join(decoded_one_sent)
    # return dec_words

    return decoded_sents


def predict(model, ot_matcher, modules, consts, options):
    print("start predicting,")
    options["has_y"] = TESTING_DATASET_CLS.HAS_Y
    if options["beam_decoding"]:
        print("using beam search")
    else:
        print("greedy search in future.")
    rebuild_dir(cfg.cc.BEAM_SUMM_PATH)
    rebuild_dir(cfg.cc.BEAM_GT_PATH)
    rebuild_dir(cfg.cc.GROUND_TRUTH_PATH)
    rebuild_dir(cfg.cc.SUMM_PATH)

    print("loading test set...")
    if options["model_selection"]:
        xyfn_list = pickle.load(open(cfg.cc.VALIDATE_DATA_PATH + "pj1000.pkl", "rb"))
    else:
        xyfn_list = pickle.load(open(cfg.cc.TESTING_DATA_PATH + "test.pkl", "rb"))
    batch_list, num_files, num_batches = data_ot.batched(len(xyfn_list), options, consts)

    print("num_files = ", num_files, ", num_batches = ", num_batches)

    google_rouge_scorer = rouge_scorer.RougeScorer(['rouge3'], use_stemmer=True)  # todo useful

    # running_start = time.time()
    partial_num = 0
    total_num = 0
    file_num = 0
    full_f_names = ''
    for idx_batch in range(num_batches):
        test_idx = batch_list[idx_batch]
        batch_raw = [xyfn_list[xyfn_idx] for xyfn_idx in test_idx]
        # print("test_idx", test_idx)

        batch = data_ot.get_data_4_ot(batch_raw, modules, consts, options, is_train=False)

        # print("len(batch.x_one_batch)", len(batch.x_one_batch))
        # print("en(test_idx)", len(test_idx))
        assert len(test_idx) == len(batch.x_one_batch)  # local_batch_size

        for idx_doc in range(len(test_idx)):
            x, len_x, x_mask, x_ext, x_sections_num, y, len_y, oy, oovs, f_names \
                = (batch.x_one_batch[idx_doc], batch.len_x_one_batch[idx_doc], batch.x_mask_one_batch[idx_doc],
                   batch.x_ext_one_batch[idx_doc], batch.x_sections_num[idx_doc],
                   batch.y_one_batch[idx_doc], batch.len_y_one_batch[idx_doc],
                   batch.original_summarys_one_batch[idx_doc], batch.x_ext_words[idx_doc],
                   batch.file_names)

            # print("x", x)
            # print("x_ext", x_ext)
            # print("x_ext", x_ext.shape)
            # print("len_x", len_x)
            # print("x_mask", x_mask)
            # print("x_mask", x_mask.shape)
            # print("x_sections_num", x_sections_num)
            # print("y", y)
            # print("len_y", len_y)
            # print("oy", oy)
            # print("oovs", oovs)

            word_emb, dec_state = model.encode(torch.LongTensor(x).to(options["device"]),
                                               torch.LongTensor(len_x).to(options["device"]),  # todo cpu
                                               torch.FloatTensor(x_mask).to(options["device"]))
            # print("word_emb", word_emb.size())
            # print("dec_state", dec_state.size())

            # encoded_sections: batch size (1) * section nums * hidden state
            encoded_sections = torch.unsqueeze(dec_state, dim=0)

            # print("encoded_sections", encoded_sections.size())

            article_sections_mask = torch.unsqueeze(torch.ones(x_sections_num).to(options["device"]), dim=0)
            # print("article_sections_mask", article_sections_mask.size())
            # print("article_sections_mask", article_sections_mask)

            masked_scaling_factors = torch.squeeze(
                ot_matcher.predict_alignment_score_4_sections(encoded_sections, [x_sections_num],
                                                              article_sections_mask), dim=0)
            # masked_scaling_factors = torch.squeeze(ot_matcher(src_articles, [x_sections_num], article_sents_mask),
            #   dim=0)
            # print("masked_scaling_factors", masked_scaling_factors.size())
            # print("masked_scaling_factors", masked_scaling_factors)
            scaling_factors_list = masked_scaling_factors.tolist()
            # print("scaling_factors_list", scaling_factors_list)

            # scaling_factors_sorted, rank_indices = masked_scaling_factors.sort(descending=True)
            # print("rank_indices", rank_indices.size(0))
            # print("rank_indices", rank_indices)
            # print("scaling_factors_sorted", scaling_factors_sorted)

            # generated_length = 0
            decoded_summary_list = []
            # for k in range(rank_indices.size(0)):
            for idx_sec, scaling_factor in enumerate(scaling_factors_list):
                # print("idx_sec", idx_sec)
                # print("scaling_factor", scaling_factor)
                decoded_sents_num_one_sec = math.floor(scaling_factor+0.5)
                # print("decoded_sents_num_one_sec", decoded_sents_num_one_sec)

                if decoded_sents_num_one_sec > 0.1:
                    if options["copy"]:
                        inputx = (torch.LongTensor(x_ext[:, idx_sec]).to(options["device"]),
                                  word_emb[:, idx_sec, :], dec_state[idx_sec, :],
                                  torch.FloatTensor(x_mask[:, idx_sec, :]).to(options["device"]),
                                  batch.max_ext_len, oovs)
                    else:
                        inputx = (torch.LongTensor(x[:, idx_sec]).to(options["device"]),
                                  word_emb[:, idx_sec, :], dec_state[idx_sec, :],
                                  torch.FloatTensor(x_mask[:, idx_sec, :]).to(options["device"]))

                    decoded_sents_one_sec = beam_decode(file_num, inputx, model, modules, consts, options,
                                                        prescribed_decoded_sents_num=decoded_sents_num_one_sec)

                    # print("num sents decoded this section: ", len(decoded_sents_one_sec))
                    # print("decoded_sents_one_sec", decoded_sents_one_sec)

                    decoded_summary_list.extend(decoded_sents_one_sec)

                # print("max_len_predict_total", consts["max_len_predict_total"])
                # if generated_length > consts["max_len_predict_total"]:
                #     print("generated_length", generated_length)
                #     break

            # todo next
            decoded_summary_list_in_str = []
            for sent in decoded_summary_list:
                decoded_summary_list_in_str.append(' '.join(sent))

            # print("decoded_summary_list_in_str", decoded_summary_list_in_str)

            decoded_summary_list_in_str = remove_repetitive_sents(decoded_summary_list_in_str, google_rouge_scorer,
                                                                  rouge_name='rouge3', remove_threshold=0.5)

            # print("len(decoded_summary_list_in_str)", len(decoded_summary_list_in_str))
            # print("filtered decoded_summary_list_in_str", decoded_summary_list_in_str)

            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            # if options["has_y"]:
            #     ly = len_y[0]
            #     y_true = y[0: ly].tolist()
            #     y_true = [str(i) for i in y_true[:-1]]  # delete <eos>

            write_for_rouge_4_ot(str(file_num), oy, decoded_summary_list_in_str, cfg)

            full_f_names = full_f_names + 'f' + str(file_num) + 'f' + ' ' + f_names[idx_doc] + '\n'
            # full_f_names.append(('f' + str(file_num) + 'f', f_names[idx_doc]))

            file_num += 1

        testing_batch_size = len(test_idx)
        partial_num += testing_batch_size
        total_num += testing_batch_size
        if partial_num >= consts["testing_print_size"]:
            print(total_num, "summs are generated")
            partial_num = 0
    print(file_num, total_num)

    name_file = "".join((cfg.cc.RESULT_PATH, 'names'))
    with open(name_file, "w") as f:
        f.write(full_f_names)


def run(existing_model_name=None):
    modules, consts, options = init_modules()
    # use_gpu(consts["idx_gpu"])
    if options["is_predicting"]:
        need_load_model = True
        training_model = False
        predict_model = True
    else:
        need_load_model = False
        training_model = True
        predict_model = False

    if options["is_continue"]:
        need_load_model = True

    print_basic_info(modules, consts, options)

    if training_model:
        print("loading train set...")
        if options["is_debugging"]:
            xyfn_list = pickle.load(open(cfg.cc.VALIDATE_DATA_PATH + "pj1000.pkl", "rb"))
        else:
            xyfn_list = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "train.pkl", "rb"))
        batch_list, num_files, num_batches = data_ot.batched(len(xyfn_list), options, consts)
        print("num_files = ", num_files, ", num_batches = ", num_batches)

    running_start = time.time()
    if True:  # TODO: refactor
        print("compiling model ...")
        model = Model4OT(modules, consts, options)
        ot_matcher = OTMatcher(modules, consts, options)

        if options["cuda"]:
            model.cuda()
            ot_matcher.cuda()  # new 4 ot

        # optimizer = torch.optim.Adagrad(model.parameters(), lr=consts["lr"], initial_accumulator_value=0.1)
        # optimizer_4_ot_matcher = torch.optim.Adam(ot_matcher.parameters(), lr=consts["lr_4_matcher"])  # new 4 ot

        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=consts["lr"], initial_accumulator_value=0.1)
        optimizer_4_ot_matcher = torch.optim.Adam(filter(lambda p: p.requires_grad, ot_matcher.parameters()),
                                                  lr=consts["lr_4_matcher"])  # new 4 ot

        model_name = "".join(["cnndm.s2s.", options["cell"]])

        if options["is_continue"]:
            existing_epoch = int(existing_model_name.split("epoch")[1].split(".")[0])
            print("Have been trained for " + str(existing_epoch) + " epochs")
            existing_epoch = existing_epoch + 1
        else:
            existing_epoch = 0

        if need_load_model:
            if existing_model_name == None:
                existing_model_name = "cnndm.s2s.gpu4.epoch7.1"
            print("loading existed model:", existing_model_name)
            # model, optimizer = load_model(cfg.cc.MODEL_PATH + existing_model_name, model, optimizer)
            model, optimizer, ot_matcher, optimizer_4_ot_matcher \
                = load_model_4_ot(cfg.cc.MODEL_PATH + existing_model_name,
                                  model, optimizer,
                                  ot_matcher, optimizer_4_ot_matcher)  # new 4 ot

        torch.autograd.set_detect_anomaly(True)

        if training_model:
            # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
            # scaler = torch.cuda.amp.GradScaler()

            print("start training model ")
            print_size = num_files // consts["print_time"] if num_files >= consts["print_time"] else num_files

            last_total_error = float("inf")
            print("max epoch:", consts["max_epoch"])

            for epoch in range(0, consts["max_epoch"]):

                num_partial = 1
                total_error = 0.0
                error_c = 0.0
                total_error_matcher = 0.0  # 4 ot

                partial_num_files = 0
                epoch_start = time.time()
                partial_start = time.time()
                # shuffle the trainset
                batch_list, num_files, num_batches = data_ot.batched(len(xyfn_list), options, consts)
                used_batch = 0.
                for idx_batch in range(num_batches):
                    train_idx = batch_list[idx_batch]
                    batch_raw = [xyfn_list[xyfn_idx] for xyfn_idx in train_idx]
                    if len(batch_raw) != consts["batch_size"]:
                        continue
                    # local_batch_size = len(batch_raw)
                    if options["pattern"] == "ot":
                        datatime = time.time()
                        batch = data_ot.get_data_4_ot(batch_raw, modules, consts, options, is_train=True)
                        # print("prepare data time:", time.time() - datatime)

                        modeltime = time.time()

                        for p in ot_matcher.parameters():
                            p.requires_grad = False

                        model.zero_grad()
                        # optimizer.zero_grad()
                        # optimizer_4_ot_matcher.zero_grad()

                        data_pack, cost, cost_c = model(batch, matcher=ot_matcher)

                        # print("model forward time:", time.time() - modeltime)

                        # print("cost", cost)
                        # print("cost_c", cost_c)

                    else:
                        print("Patterns except ot are not supported now")

                    if cost_c is None:
                        loss = cost
                    else:
                        loss = cost + cost_c
                        cost_c = cost_c.item()
                        error_c += cost_c

                    model_backwardtime = time.time()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), consts["norm_clip"])
                    optimizer.step()

                    # for p in model.parameters():
                    #     p.requires_grad = False
                    # new 4 ot todo: fix word embedding when update??
                    # print("model backward time:", time.time() - model_backwardtime)

                    matcher_forward_time = time.time()
                    # torch.cuda.empty_cache()
                    for p in ot_matcher.parameters():
                        p.requires_grad = True
                    ot_matcher.zero_grad()
                    cost_matcher = ot_matcher(data_pack, batch)
                    # print("cost_matcher", cost_matcher)

                    # print("matcher forward time:", time.time() - matcher_forward_time)

                    matcher_backwardtime = time.time()

                    cost_matcher.backward()
                    torch.nn.utils.clip_grad_norm_(ot_matcher.parameters(), consts["norm_clip"])  # TODO ??
                    optimizer_4_ot_matcher.step()

                    # for p in model.parameters():
                    #     p.requires_grad = True
                    # print("matcher backward time:", time.time() - matcher_backwardtime)

                    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

                    total_error_matcher += cost_matcher.item()

                    cost = cost.item()
                    total_error += cost
                    used_batch += 1
                    partial_num_files += consts["batch_size"]

                    # torch.cuda.empty_cache()
                    if partial_num_files // print_size == 1 and idx_batch < num_batches:
                        print(idx_batch + 1, "/", num_batches, "batches have been processed,",
                              "average cost until now:", "cost =", total_error / used_batch, ",",
                              "cost_c =", error_c / used_batch, ",",
                              "cost 4 matcher =", total_error_matcher / used_batch, ",",
                              "time:", time.time() - partial_start)
                        partial_num_files = 0
                        if not options["is_debugging"]:
                            print("save model... ", )  # save_model in utils_pg.py
                            # save_model(cfg.cc.MODEL_PATH + model_name + ".gpu" + str(consts["idx_gpu"]) + ".epoch"
                            #            + str(epoch // consts["save_epoch"] + existing_epoch) + "." + str(num_partial),
                            #            model, optimizer)
                            save_model_4_ot(cfg.cc.MODEL_PATH + model_name + ".gpu" + str(consts["idx_gpu"])
                                            + ".epoch" + str(epoch // consts["save_epoch"] + existing_epoch) +
                                            "." + str(num_partial),
                                            model, optimizer, ot_matcher, optimizer_4_ot_matcher)
                            print("finished")
                        num_partial += 1
                print("in this epoch, total average cost =", total_error / used_batch, ",",
                      "cost_c =", error_c / used_batch, ",",
                      "cost 4 matcher =", total_error_matcher / used_batch, ",",
                      "time:", time.time() - epoch_start)

                # if options["pattern"] == "none":  # TODO: refactor print for other version
                #     print_sent_dec(y_pred, y_ext, y_mask, oovs, modules, consts, options, local_batch_size)

                if last_total_error > total_error or options["is_debugging"]:
                    last_total_error = total_error
                    if not options["is_debugging"]:
                        print("save model... ", )
                        # save_model(cfg.cc.MODEL_PATH + model_name + ".gpu" + str(consts["idx_gpu"]) + ".epoch"
                        #            + str(epoch // consts["save_epoch"] + existing_epoch) + "." + str(num_partial),
                        #            model, optimizer)
                        save_model_4_ot(cfg.cc.MODEL_PATH + model_name + ".gpu" + str(consts["idx_gpu"])
                                        + ".epoch" + str(epoch // consts["save_epoch"] + existing_epoch) +
                                        "." + str(num_partial),
                                        model, optimizer, ot_matcher, optimizer_4_ot_matcher)
                        print("finished")
                else:
                    print("optimization should finished, loss begin to up !!!!!!!!!!!!!!!!")
                    # break

            print("save final model... "),
            # save_model(cfg.cc.MODEL_PATH + model_name + ".final.gpu" + str(consts["idx_gpu"]) + ".epoch"
            #            + str(epoch // consts["save_epoch"] + existing_epoch) + "." + str(num_partial),
            #            model, optimizer)
            save_model_4_ot(cfg.cc.MODEL_PATH + model_name + ".final.gpu" + str(consts["idx_gpu"]) + ".epoch"
                            + str(epoch // consts["save_epoch"] + existing_epoch) + "." + str(num_partial),
                            model, optimizer, ot_matcher, optimizer_4_ot_matcher)
            print("finished")
        else:
            print("skip training model")

        if predict_model:
            predict(model, ot_matcher, modules, consts, options)
    print("Finished, time:", time.time() - running_start)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    # TODO: refactor input format
    # existing_model_name = sys.argv[1] if len(sys.argv) > 1 else None
    existing_model_name = args.exist_model
    run(existing_model_name)
