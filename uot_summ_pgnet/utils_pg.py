# -*- coding: utf-8 -*-
# pylint: skip-file
import numpy as np
from numpy.random import random as rand
import pickle
import sys
import os
import shutil
from copy import deepcopy
import random

import torch
from torch import nn


def init_seeds():
    random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)


def init_lstm_weight(lstm):
    for param in lstm.parameters():
        if len(param.shape) >= 2:  # weights
            init_ortho_weight(param.data)
        else:  # bias
            init_bias(param.data)


def init_gru_weight(gru):
    for param in gru.parameters():
        if len(param.shape) >= 2:  # weights
            init_ortho_weight(param.data)
        else:  # bias
            init_bias(param.data)


def init_linear_weight(linear):
    init_xavier_weight(linear.weight)
    if linear.bias is not None:
        init_bias(linear.bias)


def init_normal_weight(w):
    nn.init.normal_(w, mean=0, std=0.01)


def init_uniform_weight(w):
    nn.init.uniform_(w, -0.1, 0.1)


def init_ortho_weight(w):
    nn.init.orthogonal_(w)


def init_xavier_weight(w):
    nn.init.xavier_normal_(w)


def init_bias(b):
    nn.init.constant_(b, 0.)


def rebuild_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError:
            pass
    os.mkdir(path)


def save_model(f, model, optimizer):
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
               f)


def load_model(f, model, optimizer):
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer


def sort_samples(x, len_x, mask_x, y, len_y, \
                 mask_y, oys, x_ext, y_ext, oovs):
    sorted_x_idx = np.argsort(len_x)[::-1]

    sorted_x_len = np.array(len_x)[sorted_x_idx]
    sorted_x = x[:, sorted_x_idx]
    sorted_x_mask = mask_x[:, sorted_x_idx, :]
    sorted_oovs = [oovs[i] for i in sorted_x_idx]

    sorted_y_len = np.array(len_y)[sorted_x_idx]
    sorted_y = y[:, sorted_x_idx]
    sorted_y_mask = mask_y[:, sorted_x_idx, :]
    sorted_oys = [oys[i] for i in sorted_x_idx]
    sorted_x_ext = x_ext[:, sorted_x_idx]
    sorted_y_ext = y_ext[:, sorted_x_idx]

    return sorted_x, sorted_x_len, sorted_x_mask, sorted_y, \
           sorted_y_len, sorted_y_mask, sorted_oys, \
           sorted_x_ext, sorted_y_ext, sorted_oovs


def print_sent_dec(y_pred, y, y_mask, oovs, modules, consts, options, batch_size):
    print("golden truth and prediction samples:")
    max_y_words = np.sum(y_mask, axis=0)
    max_y_words = max_y_words.reshape((batch_size))
    max_num_docs = 16 if batch_size > 16 else batch_size
    is_unicode = options["is_unicode"]
    dict_size = len(modules["i2w"])
    for idx_doc in range(max_num_docs):
        print(idx_doc + 1,
              "----------------------------------------------------------------------------------------------------")
        sent_true = ""
        for idx_word in range(max_y_words[idx_doc]):
            i = y[idx_word, idx_doc] if options["has_learnable_w2v"] else np.argmax(y[idx_word, idx_doc])
            if i in modules["i2w"]:
                sent_true += modules["i2w"][i]
            else:
                sent_true += oovs[idx_doc][i - dict_size]
            if not is_unicode:
                sent_true += " "

        if is_unicode:
            print(sent_true.encode("utf-8"))
        else:
            print(sent_true)

        print()

        sent_pred = ""
        for idx_word in range(max_y_words[idx_doc]):
            i = torch.argmax(y_pred[idx_word, idx_doc, :]).item()
            if i in modules["i2w"]:
                sent_pred += modules["i2w"][i]
            else:
                sent_pred += oovs[idx_doc][i - dict_size]
            if not is_unicode:
                sent_pred += " "
        if is_unicode:
            print(sent_pred.encode("utf-8"))
        else:
            print(sent_pred)
    print("----------------------------------------------------------------------------------------------------")
    print()


def write_for_rouge(fname, ref_sents, dec_words, cfg):
    dec_sents = []
    while len(dec_words) > 0:
        try:
            fst_period_idx = dec_words.index(".")
        except ValueError:
            fst_period_idx = len(dec_words)
        sent = dec_words[:fst_period_idx + 1]
        dec_words = dec_words[fst_period_idx + 1:]
        dec_sents.append(' '.join(sent))

    ref_file = "".join((cfg.cc.GROUND_TRUTH_PATH, fname))  # ground_truth
    decoded_file = "".join((cfg.cc.SUMM_PATH, fname))  # /summary/

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(ref_sents):
            sent = sent.strip()
            f.write(sent) if idx == len(ref_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(dec_sents):
            sent = sent.strip()
            f.write(sent) if idx == len(dec_sents) - 1 else f.write(sent + "\n")


def write_summ(dst_path, summ_list, num_summ, options, i2w=None, oovs=None, score_list=None):
    assert num_summ > 0
    with open(dst_path, "w") as f_summ:
        if num_summ == 1:
            if score_list != None:
                f_summ.write(str(score_list[0]))
                f_summ.write("\t")
            if i2w != None:
                '''
                for e in summ_list:
                    e = int(e)
                    if e in i2w:
                        print i2w[e],
                    else:
                        print oovs[e - len(i2w)],
                print "\n"
                '''
                s = []
                for e in summ_list:
                    e = int(e)
                    if e in i2w:
                        s.append(i2w[e])
                    else:
                        s.append(oovs[e - len(i2w)])
                s = " ".join(s)
            else:
                s = " ".join(summ_list)
            f_summ.write(s)
            f_summ.write("\n")
        else:
            assert num_summ == len(summ_list)
            if score_list != None:
                assert num_summ == len(score_list)

            for i in range(num_summ):
                if score_list != None:
                    f_summ.write(str(score_list[i]))
                    f_summ.write("\t")
                if i2w != None:
                    '''
                    for e in summ_list[i]:
                        e = int(e)
                        if e in i2w:
                            print i2w[e],
                        else:
                            print oovs[e - len(i2w)],
                    print "\n"
                    '''
                    s = []
                    for e in summ_list[i]:
                        e = int(e)
                        if e in i2w:
                            s.append(i2w[e])
                        else:
                            s.append(oovs[e - len(i2w)])
                    s = " ".join(s)
                else:
                    s = " ".join(summ_list[i])

                f_summ.write(s)
                f_summ.write("\n")


################## 4 ot ########################
def save_model_4_ot(f, model, optimizer_4_model, ot_matcher, optimizer_4_ot_matcher):
    torch.save({"model_state_dict": model.state_dict(),
                "ot_matcher_state_dict": ot_matcher.state_dict(),
                "optimizer_4_model_state_dict": optimizer_4_model.state_dict(),
                "optimizer_4_ot_matcher_state_dict": optimizer_4_ot_matcher.state_dict()},
               f)


def load_model_4_ot(f, model, optimizer_4_model, ot_matcher, optimizer_4_ot_matcher):
    checkpoint = torch.load(f)

    model.load_state_dict(checkpoint["model_state_dict"])
    ot_matcher.load_state_dict(checkpoint["ot_matcher_state_dict"])

    optimizer_4_model.load_state_dict(checkpoint["optimizer_4_model_state_dict"])
    optimizer_4_ot_matcher.load_state_dict(checkpoint["optimizer_4_ot_matcher_state_dict"])
    return model, optimizer_4_model, ot_matcher, optimizer_4_ot_matcher


def reshape_src_articles(src_embeddings, src_sents_num, device, pad=0.0):
    # print("src_embeddings", src_embeddings.size())
    # print("src_sents_num", sum(src_sents_num))
    # print("src_sents_num", src_sents_num)
    batch_size = len(src_sents_num)
    max_len = max(src_sents_num)
    embedding_size = src_embeddings.size(1)
    # print("embedding_size", embedding_size)

    tensor_shape = (batch_size, max_len, embedding_size)
    batched_src_embeddings = torch.FloatTensor(*tensor_shape).to(device)
    batched_src_embeddings.fill_(pad)

    src_start_idx = 0
    for idx_doc in range(batch_size):
        # print(idx_doc, src_start_idx, src_start_idx + src_sents_num[idx_doc])
        batched_src_embeddings[idx_doc, :src_sents_num[idx_doc], :] \
            = src_embeddings[src_start_idx: src_start_idx + src_sents_num[idx_doc], :]

        src_start_idx += src_sents_num[idx_doc]

    return batched_src_embeddings


def len_mask(lens, device):
    """ users are resposible for shaping
    Return: tensor_type [B, T]
    """
    max_len = max(lens)
    batch_size = len(lens)
    # mask = torch.ByteTensor(batch_size, max_len).to(device)
    mask = torch.FloatTensor(batch_size, max_len).to(device)
    mask.fill_(0.)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1.)
    return mask


# https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3
def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


# KL(x|y) = x log(x/y) -x + y
# batch size * vector length
# already masked.
def generalized_kl_div(x, y):
    # + 1e-37 to avoid nan and inf
    x = x + 1e-37
    y = y + 1e-37
    return x * (x.log() - y.log()) - x + y


def pad_rouge_list_2_tensor(rouge_list, x_nums, y_nums, device, pad=0.0):
    # print("x_nums", x_nums)
    # print("y_nums", y_nums)

    batch_size = len(rouge_list)
    max_len_x = max(x_nums)
    max_len_y = max(y_nums)
    tensor_shape = (batch_size, max_len_x, max_len_y)
    matrix_in_tensor = torch.FloatTensor(*tensor_shape).to(device)
    matrix_in_tensor.fill_(pad)
    # print("matrix_in_tensor", matrix_in_tensor)
    for i, matrix in enumerate(rouge_list):
        # print("matrix", matrix)
        matrix_in_tensor[i, :x_nums[i], :y_nums[i]] = 1 - torch.FloatTensor(matrix).to(device)

    # print("matrix_in_tensor", matrix_in_tensor)

    return matrix_in_tensor


def pad_matrix_list_2_tensor(matrix_in_list, x_nums, y_nums, device, pad=0.0):
    batch_size = len(matrix_in_list)
    max_len_x = max(x_nums)
    max_len_y = max(y_nums)
    tensor_shape = (batch_size, max_len_x, max_len_y)
    matrix_in_tensor = torch.FloatTensor(*tensor_shape).to(device)
    matrix_in_tensor.fill_(pad)
    # print("matrix_in_tensor", matrix_in_tensor)
    for i, matrix in enumerate(matrix_in_list):
        # print("matrix", matrix.size())
        # print("matrix", matrix)
        matrix_in_tensor[i, :x_nums[i], :y_nums[i]] = matrix / torch.max(matrix)

    # print("matrix_in_tensor", matrix_in_tensor.size())
    # print("matrix_in_tensor", matrix_in_tensor)
    return matrix_in_tensor

    # tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # batch_size = len(matrix_list)
    # max_len_src = max(len(matrix) for matrix in matrix_list)
    # max_len_tgt = max(len(matrix[0]) for matrix in matrix_list)
    # tensor_shape = (batch_size, max_len_src, max_len_tgt)
    # tensor = tensor_type(*tensor_shape)
    # tensor.fill_(pad)
    # for i, matrix in enumerate(matrix_list):
    #     tensor[i, :len(matrix), :len(matrix[0])] = tensor_type(matrix)
    #
    # return tensor


def handle_exception_4_cost_matrices(cost_matrices):
    if torch.isnan(cost_matrices).sum().item() > 0:  # handles NaN error
        cost_matrices = torch.where(torch.isnan(cost_matrices), torch.zeros_like(cost_matrices), cost_matrices)
    if torch.isinf(cost_matrices).sum().item() > 0:  # handles Inf error
        cost_matrices = torch.where(torch.isinf(cost_matrices), torch.zeros_like(cost_matrices), cost_matrices)

    return cost_matrices


def write_for_rouge_4_ot(fname, ref_sents, decoded_sents_list, cfg):
    # dec_sents = []
    # for decoded_sent in decoded_sents_list:
    #     dec_sents.append(' '.join(decoded_sent))

    # while len(decoded_summary_list) > 0:
    #     sent = dec_words[:fst_period_idx + 1]
    #     dec_words = dec_words[fst_period_idx + 1:]+
    #     dec_sents.append(' '.join(sent))

    ref_file = "".join((cfg.cc.GROUND_TRUTH_PATH, fname))  # ground_truth
    decoded_file = "".join((cfg.cc.SUMM_PATH, fname))  # /summary/

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(ref_sents):
            sent = sent.strip()
            f.write(sent) if idx == len(ref_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents_list):
            # sent = ' '.join(sent)
            sent = sent.strip()
            f.write(sent) if idx == len(decoded_sents_list) - 1 else f.write(sent + "\n")


def check_repetition(sent, selected_sents_list, rouge_scorer, rouge_name='rouge3'):
    if 'graphic content' in sent:
        return False

    for selected_sent in selected_sents_list:
        f_score = rouge_scorer.score(sent, selected_sent)[rouge_name][2]
        # scores = rouge_scorer.score(sent, selected_sent)[rouge_name]
        # F_score = scores[2]
        # print("scores", scores)
        # print("F_score", F_score)
        # print("*****************")

        if f_score > 0.1:
            return False
    return True


def remove_repetitive_sents(sents_list, rouge_scorer, rouge_name='rouge3', remove_threshold=0.5):
    filtered_sents_list = []

    for sent in sents_list:
        if len(filtered_sents_list) == 0:
            filtered_sents_list.append(sent)
        else:
            should_add = True
            for filtered_sent in filtered_sents_list:
                f_score = rouge_scorer.score(sent, filtered_sent)[rouge_name][2]

                if f_score > remove_threshold:
                    should_add = False
                    # print("f_score", f_score)
                    # print("sent\n", sent)
                    # print("filtered_sent\n", filtered_sent)
                    # print("*****************")
                    if len(sent) < len(filtered_sent):
                        filtered_sents_list.remove(filtered_sent)
                        filtered_sents_list.append(sent)

            if should_add:
                filtered_sents_list.append(sent)

    return filtered_sents_list


# ################# 4 ot ########################
