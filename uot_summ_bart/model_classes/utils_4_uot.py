# -*- coding: utf-8 -*-
# pylint: skip-file
import numpy as np
# from numpy.random import random as rand
# import pickle
# import sys
# from copy import deepcopy
from typing import List, Union

import os
import shutil
import random

import torch
from torch import nn

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput


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
def compute_sinkhorn_loss(cost_matrices, masked_scaling_factors, summ_side_margin,
                          epsilon=0.006, tau=0.03):
    # print("src_mask", src_mask)
    # print("tgt_mask", tgt_mask)
    # print("masked_scaling_factors", masked_scaling_factors)

    # x_y_mask = src_mask.unsqueeze(2) * tgt_mask.unsqueeze(1)

    # print("block_padding", block_padding)
    # torch.set_printoptions(profile="full")
    # print("x_y_mask", x_y_mask.size())
    # print("x_y_mask", x_y_mask)
    # print("cost_matrices 1", cost_matrices)

    # Wasserstein cost function
    # c_m = (1 - cost_matrices) * x_y_mask + (1.0 - x_y_mask) * 2.0
    # cost_matrices = cost_matrices * x_y_mask + (1.0 - x_y_mask) * block_padding
    # print("cost_matrices 2", cost_matrices)

    # positivetime = time.time()
    plan_positive = sinkhorn_batched(cost_matrices, masked_scaling_factors, summ_side_margin,
                                     is_balanced=False, epsilon=epsilon, tau=tau)
    # print("positive time:", time.time() - positivetime)

    # todo masked?
    # plan_positive = plan_positive * x_y_mask
    # print("plan_positive", plan_positive.size())
    # print("plan_positive", plan_positive)

    ot_cost_positive = torch.sum(plan_positive * cost_matrices, (1, 2))  # todo masked?

    src_marginal = torch.sum(plan_positive, 2, keepdim=False)  # x marginal

    # masked_rest_factors = torch.max(1 - masked_scaling_factors, torch.zeros_like(masked_scaling_factors)) * src_mask
    # normalized_masked_rest_factors = masked_rest_factors / torch.sum(masked_rest_factors, 1, keepdim=True)  # normalize
    # normalized_tgt_mask = tgt_mask / torch.sum(tgt_mask, 1, keepdim=True)  # normalize

    # print("normalized_masked_rest_factors", normalized_masked_rest_factors)
    # print("normalized_tgt_mask", normalized_tgt_mask)
    # negativetime = time.time()
    # plan_negative = sinkhorn_batched(cost_matrices, normalized_masked_rest_factors, normalized_tgt_mask,
    #                                  is_balanced=True, epsilon=epsilon)
    # print("negative time:", time.time() - negativetime)

    # plan_negative = plan_negative * x_y_mask
    # print("plan_negative", plan_negative)

    # ot_cost_negative = torch.sum(plan_negative * cost_matrices, (1, 2))  # todo masked?

    # print("ot_cost_positive", ot_cost_positive)
    # print("src_marginal", src_marginal)
    # print("ot_cost_negative", ot_cost_negative)

    # print("^^^^^^^^^^^^^^^^^^^^")
    # return plan_positive, ot_cost_positive, src_marginal, ot_cost_negative
    return plan_positive, ot_cost_positive, src_marginal


# batched version of unbalanced sinkhorn
def sinkhorn_batched(cost_matrix,
                     mu, nu,
                     is_balanced=False,
                     epsilon=1.0, tau=1.0, max_num_iter=300):
    # print("epsilon", epsilon)
    # print("tau", tau)

    # print("mu", mu)
    # print("nu", nu)

    mu = mu + 1e-27  # add small to avoid nan and inf
    nu = nu + 1e-27

    def M(u, v):
        # "Modified cost for logarithmic updates"
        # "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-cost_matrix + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

    def lse(A):
        # "log-sum-exp"
        return torch.log(torch.exp(A).sum(2, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err_1 = 0.0 * mu, 0.0 * nu, 0.0
    for i in range(max_num_iter):
        u1 = u  # useful to check the update
        if is_balanced:
            u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u  # balanced
        else:
            u = (epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u) * tau / (tau + epsilon)  # new 4 unbalanced
        v = epsilon * (torch.log(nu) - lse(M(u, v).transpose(1, 2)).squeeze()) + v

        # print("torch.exp(M(u, v))", torch.exp(M(u, v)))
        # err = (torch.exp(M(u, v)).sum(1, keepdim=False) - nu).abs().sum(dim=1).max().item()
        err_1 = (u - u1).abs().sum(dim=1).max().item()
        if err_1 < 1e-3:  # todo The termination threshold should be small enough!!!
            break
    # print("iteration", i)
    # print("err_1", err_1)
    pi = torch.exp(M(u, v))  # Transport plan pi = diag(a)*K*diag(b) todo: check

    if torch.isnan(pi).sum().item() > 0:  # handles NaN error
        print("Error! In this batch, certain sinkhorn optimization problem failed! Simply set the value to 0.")
        pi = torch.where(torch.isnan(pi), torch.zeros_like(pi), pi)

    # print("x marginal", torch.sum(pi, 2, keepdim=False))
    # print("y marginal", torch.sum(pi, 1, keepdim=False))
    # print("Transport plan pi", pi)

    # print("---------------")
    return pi


#  from https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def sequence_cross_entropy_with_logits_pseudo_batched(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    weights: Union[torch.FloatTensor, torch.BoolTensor],
    average: str = "batch",
    label_smoothing: float = None,
    gamma: float = None,
    alpha: Union[float, List[float], torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the `torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.
    # Parameters
    logits : `torch.FloatTensor`, required.
        A `torch.FloatTensor` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : `torch.LongTensor`, required.
        A `torch.LongTensor` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : `Union[torch.FloatTensor, torch.BoolTensor]`, required.
        A `torch.FloatTensor` of size (batch, sequence_length)
    average: `str`, optional (default = `"batch"`)
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If `None`, return a vector
        of losses per batch element.
    label_smoothing : `float`, optional (default = `None`)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like `[0.05, 0.05, 0.85, 0.05]` if the 3rd class was
        the correct label.
    gamma : `float`, optional (default = `None`)
        Focal loss[*] focusing parameter `gamma` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        `gamma` is, the more focus on hard examples.
    alpha : `Union[float, List[float]]`, optional (default = `None`)
        Focal loss[*] weighting factor `alpha` to balance between classes. Can be
        used independently with `gamma`. If a single `float` is provided, it
        is assumed binary case using `alpha` and `1 - alpha` for positive and
        negative respectively. If a list of `float` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.
    # Returns
    `torch.FloatTensor`
        A torch.FloatTensor representing the cross entropy loss.
        If `average=="batch"` or `average=="token"`, the returned loss is a scalar.
        If `average is None`, the returned loss is a vector of shape (batch_size,).
    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of None, 'token', or 'batch'")

    # make sure weights are float
    weights = weights.to(logits.dtype)
    print("weights", weights)

    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    print("non_batch_dims", non_batch_dims)

    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    print("weights_batch_sum", weights_batch_sum)

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    print("logits_flat", logits_flat.size())

    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    print("log_probs_flat", log_probs_flat.size())

    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    print("targets_flat", targets_flat.size())

    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1.0 - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    print("alpha", alpha)

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):

            # shape : (2,)
            alpha_factor = torch.tensor(
                [1.0 - float(alpha), float(alpha)], dtype=weights.dtype, device=weights.device
            )

        elif isinstance(alpha, (list, np.ndarray, torch.Tensor)):

            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)

            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(
                ("alpha must be float, list of float, or torch.FloatTensor, {} provided.").format(
                    type(alpha)
                )
            )
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(
            *targets.size()
        )
        weights = weights * alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(
            -1, targets_flat, 1.0 - label_smoothing
        )
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())

    print("negative_log_likelihood 1", negative_log_likelihood)
    # shape : (batch, sequence_length)
    # todo
    negative_log_likelihood = negative_log_likelihood * weights

    print("negative_log_likelihood 2", negative_log_likelihood.size())
    print("negative_log_likelihood 2", negative_log_likelihood)

    print("non_batch_dims", non_batch_dims)
    print("weights_batch_sum", weights_batch_sum)

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
            weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        num_non_empty_sequences = (weights_batch_sum > 0).sum() + tiny_value_of_dtype(
            negative_log_likelihood.dtype
        )
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (
            weights_batch_sum.sum() + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
            weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        print("per_batch_loss", per_batch_loss)
        return per_batch_loss


@dataclass
class Seq2SeqLMOutputWithRecombinedTargets(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    recombined_target_ids: Optional[Tuple[torch.LongTensor]] = None
    recombined_target_attention_mask: Optional[Tuple[torch.LongTensor]] = None
    src_marginal: Optional[Tuple[torch.FloatTensor]] = None


