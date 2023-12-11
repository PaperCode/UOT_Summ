# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import numpy as np
import torch
import torch as T
import torch.nn as nn
import time
from torch.autograd import Variable

from utils_pg import *
# from gru_dec import *
# from lstm_dec_v2 import *
# from lstm_dec_memory_optimized import *
from lstm_dec_4_ot import *
from unbalancedOT import compute_sinkhorn_loss

from word_prob_layer import *
import copy as copy_tool


class Model4OT(nn.Module):
    def __init__(self, modules, consts, options):
        super(Model4OT, self).__init__()

        self.dict_size = consts["dict_size"]

        ##########################################
        # print("self.dict_size", self.dict_size)
        # print("-2", modules["i2w"][self.dict_size-2])
        # print("-1", modules["i2w"][self.dict_size-1])
        # print("0", modules["i2w"][self.dict_size])
        # print("1", modules["i2w"][self.dict_size+1])
        # print("2", modules["i2w"][self.dict_size+2])

        self.pattern_type = options["pattern"]

        self._epsilon = consts["epsilon"]
        self._tau_sinkhorn = consts["tau_sinkhorn"]
        # self._tau_nn = consts["tau_nn"]

        # print("lfw_emb", modules["lfw_emb"])
        self.lfw_idx = modules["lfw_emb"]

        self._bucket_size = consts["bucket_size_4_average"]

        self._recombined_abst_max_len = consts["recombined_max_len_abst"]
        ##########################################

        self.has_learnable_w2v = options["has_learnable_w2v"]
        self.is_predicting = options["is_predicting"]
        self.is_bidirectional = options["is_bidirectional"]
        self.beam_decoding = options["beam_decoding"]
        self.cell = options["cell"]
        self.device = options["device"]
        self.copy = options["copy"]
        self.coverage = options["coverage"]
        self.avg_nll = options["avg_nll"]

        self.dim_x = consts["dim_x"]
        self.dim_y = consts["dim_y"]
        # self.len_x = consts["len_x"]
        # self.len_y = consts["len_y"]
        self.hidden_size = consts["hidden_size"]

        self.pad_token_idx = consts["pad_token_idx"]
        self.ctx_size = self.hidden_size * 2 if self.is_bidirectional else self.hidden_size

        self.w_rawdata_emb = nn.Embedding(self.dict_size, self.dim_x, self.pad_token_idx)

        if self.pattern_type == "ot":
            # TODO: just lstm version for now,  gru version  will be extended in the future
            self.encoder = nn.LSTM(self.dim_x, self.hidden_size, bidirectional=self.is_bidirectional)
            self.decoder = LSTM4OTDecoder(self.dim_y, self.hidden_size, self.ctx_size, \
                                          self.device, self.copy, self.coverage, self.is_predicting, \
                                          self.dict_size)
        else:
            if self.cell == "gru":
                self.encoder = nn.GRU(self.dim_x, self.hidden_size, bidirectional=self.is_bidirectional)
                self.decoder = GRUAttentionDecoder(self.dim_y, self.hidden_size, self.ctx_size, \
                                                   self.device, self.copy, self.coverage, self.is_predicting)
            else:
                self.encoder = nn.LSTM(self.dim_x, self.hidden_size, bidirectional=self.is_bidirectional)
                self.decoder = LSTMAttentionDecoder(self.dim_y, self.hidden_size, self.ctx_size, \
                                                    self.device, self.copy, self.coverage, self.is_predicting)

        self.get_dec_init_state = nn.Linear(self.ctx_size, self.hidden_size)

        if self.pattern_type == "none":
            self.word_prob = WordProbLayer(self.hidden_size, self.ctx_size, self.dim_y, \
                                           self.dict_size, self.device, self.copy, self.coverage)

        self.init_weights()

    def init_weights(self):
        init_uniform_weight(self.w_rawdata_emb.weight)
        if self.cell == "gru": 
            init_gru_weight(self.encoder)
        else:
            init_lstm_weight(self.encoder)
        init_linear_weight(self.get_dec_init_state)
    
    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -T.log(T.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = T.sum(cost * y_mask, 0) / T.sum(y_mask, 0)
        else:
            cost = T.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return T.mean(cost) 

    '''
    todo 2nd return value 4 unsorted??
    https://github.com/pytorch/pytorch/pull/15225
    RNNs apply sort_indices to their input hidden state and apply unsort_indices to their output hidden state.
    This is to ensure that the hidden state batches correspond to the user's ordering of input sequences.
    '''
    def encode(self, x, len_x, mask_x):
        # print("x", x.size())
        # print("len_x", len_x.size())
        # print("mask_x", mask_x.size())

        self.encoder.flatten_parameters()
        emb_x = self.w_rawdata_emb(x)
        # print("emb_x1", emb_x.data.size())
        # print("emb_x1", emb_x)
        # emb_x = torch.nn.utils.rnn.pack_padded_sequence(emb_x, len_x)
        emb_x = torch.nn.utils.rnn.pack_padded_sequence(emb_x, len_x.cpu(), enforce_sorted=False)  # todo 4 ot

        # print("emb_x", emb_x.data.size())
        # print("emb_x", emb_x)

        # hs, hn = self.encoder(emb_x, None)  # old
        hs, _ = self.encoder(emb_x, None)
        hs, _ = torch.nn.utils.rnn.pad_packed_sequence(hs)

        # print("hs", hs.size())
        dec_init_state = T.sum(hs * mask_x, 0) / T.sum(mask_x, 0)
        dec_init_state = T.tanh(self.get_dec_init_state(dec_init_state))
        return hs, dec_init_state

    def decode_once(self, y, hs, dec_init_state, mask_x, x=None, max_ext_len=None, acc_att=None):
        batch_size = hs.size(1)
        if T.sum(y) < 0:
            y_emb = Variable(T.zeros((batch_size, self.dim_y))).to(self.device)
        else:
            y_emb = self.w_rawdata_emb(y)
            y_emb = y_emb[0]
        mask_y = Variable(T.ones((batch_size, 1))).to(self.device)

        if self.copy and self.coverage:
            y_pred, hcs, C = self.decoder(y_emb, hs, dec_init_state, mask_x, \
                                          mask_y, x, acc_att, max_ext_len=max_ext_len)
        elif self.copy:
            y_pred, hcs, C = self.decoder(y_emb, hs, dec_init_state, mask_x, \
                                          mask_y, xid=x, max_ext_len=max_ext_len)
        elif self.coverage:
            y_pred, hcs, C = self.decoder(y_emb, hs, dec_init_state, mask_x, \
                                          mask_y, init_coverage=acc_att, max_ext_len=max_ext_len)
        else:
            y_pred, hcs, C = self.decoder(y_emb, hs, dec_init_state, mask_x, \
                                          mask_y, max_ext_len=max_ext_len)

        if self.coverage:
            return y_pred, hcs, C
        else:
            return y_pred, hcs

    def decode_once_old(self, y, hs, dec_init_state, mask_x, x=None, max_ext_len=None, acc_att=None):
        batch_size = hs.size(1)
        if T.sum(y) < 0:
            y_emb = Variable(T.zeros((1, batch_size, self.dim_y))).to(self.device)
        else:
            y_emb = self.w_rawdata_emb(y)
        mask_y = Variable(T.ones((1, batch_size, 1))).to(self.device)

        if self.copy and self.coverage:
            hcs, dec_status, atted_context, att_dist, xids, C \
                = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, x, acc_att)
        elif self.copy:
            hcs, dec_status, atted_context, att_dist, xids \
                = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, xid=x)
        elif self.coverage:
            hcs, dec_status, atted_context, att_dist, C \
                = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, init_coverage=acc_att)
        else:
            hcs, dec_status, atted_context \
                = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y)
        
        if self.copy:
            y_pred = self.word_prob(dec_status, atted_context, y_emb, att_dist, xids, max_ext_len)
        else:
            y_pred = self.word_prob(dec_status, atted_context, y_emb)

        if self.coverage:
            return y_pred, hcs, C
        else:
            return y_pred, hcs

    # todo if one word in y_ext is not in x_ext of the same pair, set it to oov
    def handle_oovs_dynamically(self, x_ext_tensor, y_ext_tensor):
        x_ext_tensor = torch.transpose(x_ext_tensor, 0, 1)
        y_ext_tensor = torch.transpose(y_ext_tensor, 0, 1)

        # print("y_ext_tensor", y_ext_tensor.size())
        select_mask = torch.ge(y_ext_tensor, self.dict_size).float()   # get index
        # print("select_mask", select_mask)

        pair_idx = torch.nonzero(select_mask, as_tuple=False)
        # print("pair_idx", pair_idx)

        for i in range(pair_idx.size(0)):
            if y_ext_tensor[pair_idx[i, 0], pair_idx[i, 1]] not in x_ext_tensor[pair_idx[i, 0]]:
                y_ext_tensor[pair_idx[i, 0], pair_idx[i, 1]] = self.lfw_idx

        return torch.transpose(y_ext_tensor, 0, 1)

    # todo independent of expansion in  construct_cost_with_loss
    def filter_s2s_pairs(self, alignment_plans, x_sents_nums,
                         mask_x_list, x_ext_list, hs_all, h0_all,
                         y_list, y_ext_list, y_mask_list, max_ext_len):
        # print("hs_all", hs_all.size())
        # print("h0_all", h0_all.size())

        scores = []

        # src side
        expanded_mask_x = []
        expanded_x_ext = []

        expanded_hs = []
        expanded_h0 = []

        # tgt side
        expanded_y = []
        expanded_y_ext = []
        expanded_y_mask = []

        src_start_idx = 0
        for idx_doc in range(len(x_sents_nums)):
            # print("src_start_idx", src_start_idx)

            plan = alignment_plans[idx_doc]
            # print("plan", plan)
            # print("plan", plan.size())

            # print("plan", torch.gt(plan, 1e-5))

            # torch.nonzero: The result is sorted lexicographically, with the last index changing the fastest (C-style).
            select_mask = torch.gt(plan, 1e-1)  # get index
            # print("select_mask", select_mask)

            # reshape, note the order?
            # https://pytorch.org/docs/stable/generated/torch.masked_select.html#torch.masked_select
            alignment_score = torch.masked_select(plan, select_mask)  # src 1st, tgt 2nd
            # print("alignment_score", alignment_score)
            scores.append(alignment_score)

            pair_idx = torch.nonzero(select_mask)
            # print("pair_idx 0 ", pair_idx)
            # print("pair_idx", pair_idx[:, 0])
            # print("pair_idx", pair_idx[:, 1])

            selection_mask_4_x = pair_idx[:, 0]
            selection_mask_4_y = pair_idx[:, 1]

            # src side
            mask_x = torch.FloatTensor(mask_x_list[idx_doc]).to(self.device)
            # print("mask_x", mask_x.size())
            # print("mask_x", mask_x)
            mask_x = torch.index_select(mask_x, dim=1, index=selection_mask_4_x)
            # print("mask_x 2", mask_x.size())
            expanded_mask_x.append(mask_x)

            x_ext = torch.LongTensor(x_ext_list[idx_doc]).to(self.device)
            # print("x_ext", x_ext.size())
            x_ext = torch.index_select(x_ext, dim=1, index=selection_mask_4_x)
            expanded_x_ext.append(x_ext)

            hs = torch.index_select(hs_all, dim=1, index=(selection_mask_4_x+src_start_idx))
            # print("selection_mask_4_x+src_start_idx", selection_mask_4_x+src_start_idx)
            # print("hs", hs.size())
            expanded_hs.append(hs)

            h0 = torch.index_select(h0_all, dim=0, index=(selection_mask_4_x+src_start_idx))
            # print("h0", h0.size())

            expanded_h0.append(h0)

            # tgt side
            y = torch.LongTensor(y_list[idx_doc]).to(self.device)
            # print("y", y.size())
            y = torch.index_select(y, dim=1, index=selection_mask_4_y)
            # print("y", y)
            expanded_y.append(y)

            y_ext = torch.LongTensor(y_ext_list[idx_doc]).to(self.device)
            # print("y_ext", y_ext.size())
            y_ext = torch.index_select(y_ext, dim=1, index=selection_mask_4_y)
            expanded_y_ext.append(y_ext)

            y_mask = torch.FloatTensor(y_mask_list[idx_doc]).to(self.device)
            # print("y_mask", y_mask.size())
            y_mask = torch.index_select(y_mask, dim=1, index=selection_mask_4_y)
            expanded_y_mask.append(y_mask)

            src_start_idx += x_sents_nums[idx_doc]
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        scores = torch.cat(scores, dim=0)
        # print("scores", scores)

        expanded_mask_x = T.cat(expanded_mask_x, dim=1)
        expanded_x_ext = T.cat(expanded_x_ext, dim=1)

        expanded_hs = T.cat(expanded_hs, dim=1)
        expanded_h0 = T.cat(expanded_h0, dim=0)
        # hs_all.detach_()
        # h0_all.detach_()

        expanded_y = T.cat(expanded_y, dim=1)
        expanded_y_ext = T.cat(expanded_y_ext, dim=1)
        expanded_y_mask = T.cat(expanded_y_mask, dim=1)

        expanded_y_ext = self.handle_oovs_dynamically(expanded_x_ext, expanded_y_ext)

        # print("expanded_mask_x", expanded_mask_x.size())
        # print("expanded_x_ext", expanded_x_ext.size())
        # print("expanded_hs", expanded_hs.size())
        # print("expanded_h0", expanded_h0.size())
        # print("expanded_y", expanded_y.size())
        # print("expanded_y_ext", expanded_y_ext.size())
        # print("expanded_y_mask", expanded_y_mask.size())

        # y_emb = self.w_rawdata_emb(y)
        # y_shifted = y_emb[:-1, :, :]  # todo : check eos
        y_shifted = (self.w_rawdata_emb(expanded_y))[:-1, :, :]
        y_shifted = T.cat((Variable(torch.zeros(1, *y_shifted[0].size())).to(self.device), y_shifted), 0)

        # h0 = dec_init_state
        if self.cell == "lstm":
            expanded_h0 = (expanded_h0, expanded_h0)
        if self.coverage:  # todo : check shape: B * len(x)
            acc_att = Variable(torch.zeros(T.transpose(expanded_x_ext, 0, 1).size())).to(self.device)
            # print("acc_att", acc_att.size())

        if self.copy and self.coverage:
            cost, cost_c = self.decoder(y_shifted, expanded_hs, expanded_h0, expanded_mask_x, expanded_y_mask, \
                                        expanded_x_ext, acc_att, max_ext_len, expanded_y, expanded_y_ext, self.avg_nll)
            # cost = cost + cost_c
        elif self.copy:  # TODO: refactor  att_dist ?
            cost = self.decoder(y_shifted, expanded_hs, expanded_h0, expanded_mask_x, expanded_y_mask, \
                                xid=expanded_x_ext, max_ext_len=max_ext_len, y_idx=expanded_y, y_ext_idx=expanded_y_ext,
                                use_avg_nll=self.avg_nll)
        elif self.coverage:
            cost, cost_c = self.decoder(y_shifted, expanded_hs, expanded_h0, expanded_mask_x, expanded_y_mask, \
                                        init_coverage=acc_att, y_idx=expanded_y, y_ext_idx=expanded_y_ext,
                                        use_avg_nll=self.avg_nll)
            # cost = cost + cost_c
        else:
            cost = self.decoder(y_shifted, expanded_hs, expanded_h0, expanded_mask_x, expanded_y_mask, \
                                y_idx=expanded_y, y_ext_idx=expanded_y_ext, use_avg_nll=self.avg_nll)

        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("cost_c 1", cost_c.size())
        # print("cost_c 1", cost_c)
        # print("scores", scores.size())
        # print("scores", torch.sum(scores))
        # print("cost xxx", cost.size())
        # print("cost_c 1", cost_c)

        cost = torch.sum(cost * scores) / torch.sum(scores)
        if self.coverage:
            cost_c = torch.sum(cost_c * scores) / torch.sum(scores)

            # print("cost", cost)
            return cost, cost_c
        else:
            return cost, None

    @staticmethod
    @torch.no_grad()
    def get_section_summary_alignment_index(alignment_plans_one_batch, x_sections_num, y_sents_nums):
        # print("x_sections_num", x_sections_num)
        # print("y_sents_nums", y_sents_nums)
        # print("alignment_plans_one_batch", alignment_plans_one_batch.size())
        # print("alignment_plans_one_batch", alignment_plans_one_batch)

        aligned_sections_index = torch.argmax(alignment_plans_one_batch, dim=1)

        # print("aligned_sections_index", aligned_sections_index.size())
        # print("aligned_sections_index", aligned_sections_index)

        # [[(section index, (summ sentence indices))]]
        section_summary_indices = []
        for doc_idx, sec_num in enumerate(x_sections_num):
            indices_of_one_doc = []
            aligned_sections = aligned_sections_index[doc_idx, :y_sents_nums[doc_idx]]
            # print("aligned_sections", aligned_sections)
            for sec_idx in range(sec_num):
                aligned_summ_sent_indices = \
                    torch.nonzero((aligned_sections == sec_idx).float(), as_tuple=True)[0].tolist()
                if len(aligned_summ_sent_indices) > 0:
                    indices_of_one_doc.append((sec_idx, aligned_summ_sent_indices))

            section_summary_indices.append(indices_of_one_doc)
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        # print("section_summary_indices", section_summary_indices)
        assert len(section_summary_indices) == len(x_sections_num)
        return section_summary_indices

    def filter_and_recombine_section_summary_alignment_pairs(
            self, section_summary_indices, x_sections_num,
            mask_x_list, x_ext_list, hs_all, h0_all,
            y_list, y_ext_list, y_mask_list, y_len_list):
        # print("self._recombined_abst_max_len", self._recombined_abst_max_len)
        # torch.set_printoptions(profile="full")
        # print("hs_all", hs_all.size())
        # print("h0_all", h0_all.size())

        # print("y_list", y_list)
        # print("y_ext_list", y_ext_list)
        # print("y_mask_list", y_mask_list)
        # print("y_len_list", y_len_list)
        # print("section_summary_indices", section_summary_indices)
        # print("x_sections_num", x_sections_num)

        # src side
        recombined_mask_x = []
        recombined_x_ext = []

        recombined_hs = []
        recombined_h0 = []

        # tgt side
        recombined_y = []
        recombined_y_ext = []
        recombined_y_mask = []

        src_start_idx = 0
        recombined_y_len_one_batch = []

        for idx_doc in range(len(x_sections_num)):

            selection_mask_4_x, selection_mask_4_y = zip(*(section_summary_indices[idx_doc]))

            num_pairs_one_doc = len(list(selection_mask_4_x))
            selection_mask_4_x = torch.LongTensor(list(selection_mask_4_x)).to(self.device)
            selection_mask_4_y = list(selection_mask_4_y)
            # print("num_pairs_one_doc", num_pairs_one_doc)
            # print("selection_mask_4_x", selection_mask_4_x)
            # print("selection_mask_4_y", selection_mask_4_y)

            # src side
            mask_x = torch.FloatTensor(mask_x_list[idx_doc]).to(self.device)
            # print("mask_x", mask_x.size())
            # print("mask_x", mask_x)
            mask_x = torch.index_select(mask_x, dim=1, index=selection_mask_4_x)
            # print("mask_x 2", mask_x.size())
            recombined_mask_x.append(mask_x)

            x_ext = torch.LongTensor(x_ext_list[idx_doc]).to(self.device)
            # print("x_ext", x_ext.size())
            # print("x_ext", x_ext)
            x_ext = torch.index_select(x_ext, dim=1, index=selection_mask_4_x)
            # print("x_ext", x_ext.size())
            # print("x_ext", x_ext)
            recombined_x_ext.append(x_ext)

            hs = torch.index_select(hs_all, dim=1, index=(selection_mask_4_x + src_start_idx))
            # print("selection_mask_4_x+src_start_idx", selection_mask_4_x + src_start_idx)
            # print("hs", hs.size())
            recombined_hs.append(hs)

            h0 = torch.index_select(h0_all, dim=0, index=(selection_mask_4_x + src_start_idx))
            # print("h0", h0.size())
            recombined_h0.append(h0)

            # tgt side
            # need transpose
            y = torch.zeros(self._recombined_abst_max_len, num_pairs_one_doc, dtype=torch.long).to(self.device)
            y_ext = torch.zeros(self._recombined_abst_max_len, num_pairs_one_doc, dtype=torch.long).to(self.device)
            y_mask = torch.zeros(self._recombined_abst_max_len, num_pairs_one_doc, 1, dtype=torch.float).to(self.device)

            y_len_one_doc = y_len_list[idx_doc]
            # print("y_len_one_doc", y_len_one_doc)
            for sec_idx, summ_idx_4_sec in enumerate(selection_mask_4_y):
                # print("sec_idx", sec_idx)
                # print("summ_idx_4_sec", summ_idx_4_sec)
                acculumated_len_y = 0
                for summ_sen_idx in summ_idx_4_sec:
                    if acculumated_len_y + y_len_one_doc[summ_sen_idx] < self._recombined_abst_max_len:
                        y[acculumated_len_y: acculumated_len_y + y_len_one_doc[summ_sen_idx], sec_idx] \
                            = torch.LongTensor(y_list[idx_doc][0:y_len_one_doc[summ_sen_idx], summ_sen_idx])
                        y_ext[acculumated_len_y: acculumated_len_y + y_len_one_doc[summ_sen_idx], sec_idx] \
                            = torch.LongTensor(y_ext_list[idx_doc][0:y_len_one_doc[summ_sen_idx], summ_sen_idx])
                        y_mask[acculumated_len_y: acculumated_len_y + y_len_one_doc[summ_sen_idx], sec_idx, 0] \
                            = torch.FloatTensor(y_mask_list[idx_doc][0:y_len_one_doc[summ_sen_idx], summ_sen_idx, 0])

                        acculumated_len_y = acculumated_len_y + y_len_one_doc[summ_sen_idx]
                        # print("acculumated_len_y", acculumated_len_y)
                recombined_y_len_one_batch.append(acculumated_len_y)
                # print("++++++++++++++++++++++++++++++++")

            # print("y_list[idx_doc]", y_list[idx_doc].shape)
            # print("y_list[idx_doc]", y_list[idx_doc])
            # print("y", y.size())
            # print("y", y)
            recombined_y.append(y)

            # print("y_ext_list[idx_doc]", y_ext_list[idx_doc].shape)
            # print("y_ext_list[idx_doc]", y_ext_list[idx_doc])
            # print("y_ext", y_ext.size())
            # print("y_ext", y_ext)
            recombined_y_ext.append(y_ext)

            # print("y_mask_list[idx_doc]", y_mask_list[idx_doc].shape)
            # print("y_mask_list[idx_doc]", y_mask_list[idx_doc][:, :, 0])
            # print("y_mask", y_mask.size())
            # print("y_mask", y_mask[:, :, 0])
            recombined_y_mask.append(y_mask)

            # print("src_start_idx", src_start_idx)
            src_start_idx += x_sections_num[idx_doc]
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        recombined_mask_x = T.cat(recombined_mask_x, dim=1)
        recombined_x_ext = T.cat(recombined_x_ext, dim=1)

        recombined_hs = T.cat(recombined_hs, dim=1)
        recombined_h0 = T.cat(recombined_h0, dim=0)

        recombined_y = T.cat(recombined_y, dim=1)
        recombined_y_ext = T.cat(recombined_y_ext, dim=1)
        recombined_y_mask = T.cat(recombined_y_mask, dim=1)

        assert len(recombined_y_len_one_batch) == recombined_x_ext.size(1)
        # print("recombined_y_len_one_batch", recombined_y_len_one_batch)
        recombined_max_y_length = max(recombined_y_len_one_batch)
        # print("recombined_max_y_length", recombined_max_y_length)

        recombined_y = recombined_y[0:recombined_max_y_length, :]
        recombined_y_ext = recombined_y_ext[0:recombined_max_y_length, :]
        recombined_y_mask = recombined_y_mask[0:recombined_max_y_length, :, :]
        # print("recombined_y", recombined_y.size())
        # print("recombined_y_ext", recombined_y_ext.size())
        # print("recombined_y_mask", recombined_y_mask.size())
        # print("recombined_y_mask", recombined_y_mask[:, :, 0])

        recombined_y_ext = self.handle_oovs_dynamically(recombined_x_ext, recombined_y_ext)

        return recombined_mask_x, recombined_x_ext,\
            recombined_hs, recombined_h0,\
            recombined_y, recombined_y_ext, recombined_y_mask

    # todo independent of expansion in  construct_cost_with_loss
    def filter_and_recombine_s2s(self, alignment_plans, x_sections_num, y_sents_nums,
                                 mask_x_list, x_ext_list, hs_all, h0_all,
                                 y_list, y_ext_list, y_mask_list, y_len_list, max_ext_len):
        # aligntime = time.time()
        section_summary_indices = self.get_section_summary_alignment_index(
            alignment_plans.clone().detach(), x_sections_num, y_sents_nums)
        # print("section summary alignment time:", time.time() - aligntime)

        # recombinetime = time.time()
        recombined_mask_x, recombined_x_ext, recombined_hs, recombined_h0, \
            recombined_y, recombined_y_ext, recombined_y_mask = \
            self.filter_and_recombine_section_summary_alignment_pairs(
                section_summary_indices, x_sections_num,
                mask_x_list, x_ext_list, hs_all, h0_all,
                y_list, y_ext_list, y_mask_list, y_len_list)
        # print("recombine data time:", time.time() - recombinetime)

        # y_emb = self.w_rawdata_emb(y)
        # y_shifted = y_emb[:-1, :, :]  # todo : check eos
        y_shifted = (self.w_rawdata_emb(recombined_y))[:-1, :, :]
        y_shifted = T.cat((Variable(torch.zeros(1, *y_shifted[0].size())).to(self.device), y_shifted), 0)

        if torch.isnan(y_shifted).sum().item() > 0:  # handles NaN error
            print("y_shifted contains Nan.")
            y_shifted = torch.where(torch.isnan(y_shifted), torch.zeros_like(y_shifted), y_shifted)
        if torch.isnan(recombined_hs).sum().item() > 0:  # handles NaN error
            print("recombined_hs contains Nan.")
            recombined_hs = torch.where(torch.isnan(recombined_hs), torch.zeros_like(recombined_hs), recombined_hs)
        if torch.isnan(recombined_h0).sum().item() > 0:  # handles NaN error
            print("recombined_h0 contains Nan.")
            recombined_h0 = torch.where(torch.isnan(recombined_h0), torch.zeros_like(recombined_h0), recombined_h0)
        if torch.isnan(recombined_mask_x).sum().item() > 0:  # handles NaN error
            print("recombined_mask_x contains Nan.")
            recombined_mask_x = torch.where(torch.isnan(recombined_mask_x), torch.zeros_like(recombined_mask_x), recombined_mask_x)
        if torch.isnan(recombined_y_mask).sum().item() > 0:  # handles NaN error
            print("recombined_y_mask contains Nan.")
            recombined_y_mask = torch.where(torch.isnan(recombined_y_mask), torch.zeros_like(recombined_y_mask), recombined_y_mask)
        if torch.isnan(recombined_x_ext).sum().item() > 0:  # handles NaN error
            print("recombined_x_ext contains Nan.")
            recombined_x_ext = torch.where(torch.isnan(recombined_x_ext), torch.zeros_like(recombined_x_ext), recombined_x_ext)
        if torch.isnan(recombined_y).sum().item() > 0:  # handles NaN error
            print("recombined_y contains Nan.")
            recombined_y = torch.where(torch.isnan(recombined_y), torch.zeros_like(recombined_y), recombined_y)
        if torch.isnan(recombined_y_ext).sum().item() > 0:  # handles NaN error
            print("recombined_y_ext contains Nan.")
            recombined_y_ext = torch.where(torch.isnan(recombined_y_ext), torch.zeros_like(recombined_y_ext), recombined_y_ext)

        # truedecodetime = time.time()
        # h0 = dec_init_state
        if self.cell == "lstm":
            recombined_h0 = (recombined_h0, recombined_h0)
        if self.coverage:  # todo : check shape: B * len(x)
            acc_att = Variable(torch.zeros(T.transpose(recombined_x_ext, 0, 1).size())).to(self.device)
            # print("acc_att", acc_att.size())

        if self.copy and self.coverage:
            cost, cost_c = self.decoder(y_shifted, recombined_hs, recombined_h0, recombined_mask_x, recombined_y_mask, \
                                        recombined_x_ext, acc_att, max_ext_len, recombined_y, recombined_y_ext, self.avg_nll)
            # cost = cost + cost_c
        elif self.copy:  # TODO: refactor  att_dist ?
            cost = self.decoder(y_shifted, recombined_hs, recombined_h0, recombined_mask_x, recombined_y_mask, \
                                xid=recombined_x_ext, max_ext_len=max_ext_len, y_idx=recombined_y, y_ext_idx=recombined_y_ext,
                                use_avg_nll=self.avg_nll)
        elif self.coverage:
            cost, cost_c = self.decoder(y_shifted, recombined_hs, recombined_h0, recombined_mask_x, recombined_y_mask, \
                                        init_coverage=acc_att, y_idx=recombined_y, y_ext_idx=recombined_y_ext,
                                        use_avg_nll=self.avg_nll)
            # cost = cost + cost_c
        else:
            cost = self.decoder(y_shifted, recombined_hs, recombined_h0, recombined_mask_x, recombined_y_mask, \
                                y_idx=recombined_y, y_ext_idx=recombined_y_ext, use_avg_nll=self.avg_nll)

        # print("true decode time:", time.time() - truedecodetime)

        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("cost_c 1", cost_c.size())
        # print("cost_c 1", cost_c)
        # print("cost xxx", cost.size())
        # print("cost xxx", cost)

        if self.coverage:
            # print("cost", cost)
            return cost.mean(), cost_c.mean()
        else:
            return cost.mean(), None

    @torch.no_grad()
    def construct_cost_with_loss(self, x_sections_num, y_sents_nums,
                                 mask_x_list, x_ext_list, hs_all, h0_all,
                                 y_list, y_ext_list, y_mask_list, max_ext_len):  # 4 later stage
        print("x_sections_num", x_sections_num)
        print("y_sents_nums", y_sents_nums)
        # torch.set_printoptions(profile="full")

        cost_matrix_list = []

        src_start_idx = 0
        for idx_doc in range(len(x_sections_num)):
            # print("src_start_idx", src_start_idx)

            # print("x_sents_nums[idx_doc]", x_sents_nums[idx_doc])
            # print("y_sents_nums[idx_doc]", y_sents_nums[idx_doc])

            # src side
            mask_x = torch.FloatTensor(mask_x_list[idx_doc]).to(self.device)
            # print("mask_x 1", mask_x.size())
            mask_x = mask_x.repeat(1, y_sents_nums[idx_doc], 1)
            # print("mask_x 2", mask_x.size())

            x_ext = torch.LongTensor(x_ext_list[idx_doc]).to(self.device)
            # print("x_ext", x_ext.size())
            x_ext = x_ext.repeat(1, y_sents_nums[idx_doc])
            # print("x_ext", x_ext.size())
            # print("x_ext", x_ext)

            # print("hs_all", hs_all.size())
            hs = hs_all[:, src_start_idx:src_start_idx + x_sections_num[idx_doc], :]
            hs = hs.repeat(1, y_sents_nums[idx_doc], 1)
            # print("hs", hs.size())

            # print("h0_all", h0_all.size())
            h0 = h0_all[src_start_idx:src_start_idx + x_sections_num[idx_doc], :]
            h0 = h0.repeat(y_sents_nums[idx_doc], 1)
            # print("h0", h0.size())

            # tgt side
            y = torch.LongTensor(y_list[idx_doc]).to(self.device)
            # print("y", y.size())
            y = torch.repeat_interleave(y, x_sections_num[idx_doc], dim=1)
            # print("y", y.size())
            # print("y", y)

            y_ext = torch.LongTensor(y_ext_list[idx_doc]).to(self.device)
            # print("y_ext", y_ext.size())
            y_ext = torch.repeat_interleave(y_ext, x_sections_num[idx_doc], dim=1)
            # print("y_ext", y_ext.size())
            # print("y_ext", y_ext)

            y_mask = torch.FloatTensor(y_mask_list[idx_doc]).to(self.device)
            # print("y_mask", y_mask.size())
            y_mask = torch.repeat_interleave(y_mask, x_sections_num[idx_doc], dim=1)
            # print("y_mask", y_mask.size())
            # print("y_mask", y_mask)

            y_ext = self.handle_oovs_dynamically(x_ext, y_ext)  # todo

            # y_emb = self.w_rawdata_emb(y)
            # y_shifted = y_emb[:-1, :, :]  # todo : check eos
            y_shifted = (self.w_rawdata_emb(y))[:-1, :, :]
            y_shifted = T.cat((Variable(torch.zeros(1, *y_shifted[0].size())).to(self.device), y_shifted), 0)

            # h0 = dec_init_state
            if self.cell == "lstm":
                h0 = (h0, h0)
            if self.coverage:  # todo : check shape: B * len(x)
                acc_att = Variable(torch.zeros(T.transpose(x_ext, 0, 1).size())).to(self.device)
                # print("acc_att", acc_att.size())

            # print("y_shifted", y_shifted.size())

            if self.copy and self.coverage:
                cost, cost_c = self.decoder(y_shifted, hs, h0, mask_x, y_mask, \
                                            x_ext, acc_att, max_ext_len, y, y_ext, self.avg_nll)
                cost = cost + cost_c
            elif self.copy:  # TODO: refactor  att_dist ?
                cost = self.decoder(y_shifted, hs, h0, mask_x, y_mask, \
                                    xid=x_ext, max_ext_len=max_ext_len, y_idx=y, y_ext_idx=y_ext,
                                    use_avg_nll=self.avg_nll)
            elif self.coverage:
                cost, cost_c = self.decoder(y_shifted, hs, h0, mask_x, y_mask, \
                                            init_coverage=acc_att, y_idx=y, y_ext_idx=y_ext,
                                            use_avg_nll=self.avg_nll)
                cost = cost + cost_c
            else:
                cost = self.decoder(y_shifted, hs, h0, mask_x, y_mask, \
                                    y_idx=y, y_ext_idx=y_ext, use_avg_nll=self.avg_nll)

            # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
            # cost_without_chain = cost.clone().detach()

            # todo no gradient
            # print("cost 1", cost.size())
            # print("cost 1", cost)
            cost = torch.transpose(cost.view(y_sents_nums[idx_doc], x_sections_num[idx_doc]), 0, 1)  # todo check sequence
            # print("cost 2", cost.size())
            # print("cost 2", cost)

            cost_matrix_list.append(cost.clone())

            del cost  # 4 save memory
            del mask_x
            del x_ext
            del hs
            del h0
            del y
            del y_ext
            del y_mask
            del y_shifted
            del acc_att

            # torch.cuda.empty_cache()

            src_start_idx += x_sections_num[idx_doc]

            # print("idx_doc", idx_doc)
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        return cost_matrix_list

        # cost.detach_()
        # mask_x.detach_()
        # x_ext.detach_()
        # hs.detach_()
        # h0.detach_()
        # y.detach_()
        # y_ext.detach_()
        # y_mask.detach_()
        # y_shifted.detach_()
        # acc_att.detach_()

    # https://discuss.pytorch.org/t/runtimeerror-expected-scalar-type-half-but-found-float/91628/3
    # https://github.com/pytorch/pytorch/issues/42605
    # @torch.cuda.amp.autocast()
    @torch.no_grad()
    def construct_cost_with_loss_bucket_version(self, x_sections_num, y_sents_nums,
                                                mask_x_list, x_ext_list, hs_all, h0_all,
                                                y_list, y_ext_list, y_mask_list, max_ext_len):  # 4 later stage
        # torch.set_printoptions(profile="full")
        # print("self._bucket_size", self._bucket_size)
        # print("x_sections_num", x_sections_num)
        # print("y_sents_nums", y_sents_nums)

        all_buckets = []

        accumulated = 0
        one_bucket = []
        for idx in range(len(x_sections_num)):
            if accumulated + x_sections_num[idx] * y_sents_nums[idx] < self._bucket_size:
                one_bucket.append(idx)
                accumulated = accumulated + x_sections_num[idx] * y_sents_nums[idx]
            else:
                if idx > 0:
                    all_buckets.append(one_bucket)
                accumulated = x_sections_num[idx] * y_sents_nums[idx]
                one_bucket = [idx]
            # print("idx, one_bucket, accumulated", idx, one_bucket, accumulated)

        all_buckets.append(one_bucket)
        # print("all_buckets", all_buckets)

        recovered_indices = [idx for bu in all_buckets for idx in bu]
        assert recovered_indices == list(range(len(x_sections_num)))

        # print("+++++++++++++++++++++++++++++++++++")
        cost_matrix_list = []
        src_start_idx = 0

        decoder_low_precision = copy_tool.deepcopy(self.decoder).bfloat16()

        for one_bucket in all_buckets:
            # print("one_bucket", one_bucket)
            # src side
            mask_x_one_bucket = []
            x_ext_one_bucket = []

            hs_one_bucket = []
            h0_one_bucket = []

            # tgt side
            y_one_bucket = []
            y_ext_one_bucket = []
            y_mask_one_bucket = []

            for idx_doc in one_bucket:
                # print("idx_doc", idx_doc)
                # print("src_start_idx", src_start_idx)

                # print("x_sections_num[idx_doc]", x_sections_num[idx_doc])
                # print("y_sents_nums[idx_doc]", y_sents_nums[idx_doc])

                # src side
                mask_x = torch.BFloat16Tensor(mask_x_list[idx_doc]).to(self.device)
                # print("mask_x 1", mask_x.size())
                mask_x = mask_x.repeat(1, y_sents_nums[idx_doc], 1)
                # print("mask_x 2", mask_x.size())
                mask_x_one_bucket.append(mask_x)

                x_ext = torch.LongTensor(x_ext_list[idx_doc]).to(self.device)
                # print("x_ext", x_ext.size())
                x_ext = x_ext.repeat(1, y_sents_nums[idx_doc])
                # print("x_ext", x_ext.size())
                # print("x_ext", x_ext)
                x_ext_one_bucket.append(x_ext)

                # print("hs_all", hs_all.size())
                hs = hs_all[:, src_start_idx:src_start_idx + x_sections_num[idx_doc], :]
                hs = hs.repeat(1, y_sents_nums[idx_doc], 1)
                # print("hs", hs.size())
                hs_one_bucket.append(hs)

                # print("h0_all", h0_all.size())
                h0 = h0_all[src_start_idx:src_start_idx + x_sections_num[idx_doc], :]
                h0 = h0.repeat(y_sents_nums[idx_doc], 1)
                # print("h0", h0.size())
                h0_one_bucket.append(h0)

                # tgt side
                y = torch.LongTensor(y_list[idx_doc]).to(self.device)
                # print("y", y.size())
                y = torch.repeat_interleave(y, x_sections_num[idx_doc], dim=1)
                # print("y", y.size())
                # print("y", y)
                y_one_bucket.append(y)

                y_ext = torch.LongTensor(y_ext_list[idx_doc]).to(self.device)
                # print("y_ext", y_ext.size())
                y_ext = torch.repeat_interleave(y_ext, x_sections_num[idx_doc], dim=1)
                # print("y_ext", y_ext.size())
                # print("y_ext", y_ext)
                y_ext_one_bucket.append(y_ext)

                y_mask = torch.BFloat16Tensor(y_mask_list[idx_doc]).to(self.device)
                # print("y_mask", y_mask.size())
                y_mask = torch.repeat_interleave(y_mask, x_sections_num[idx_doc], dim=1)
                # print("y_mask", y_mask.size())
                # print("y_mask", y_mask)
                y_mask_one_bucket.append(y_mask)

                src_start_idx += x_sections_num[idx_doc]
                # print("+++++++++++++++++++++++++++++++++++")

            mask_x_one_bucket = T.cat(mask_x_one_bucket, dim=1)
            x_ext_one_bucket = T.cat(x_ext_one_bucket, dim=1)

            hs_one_bucket = T.cat(hs_one_bucket, dim=1)
            h0_one_bucket = T.cat(h0_one_bucket, dim=0)

            y_one_bucket = T.cat(y_one_bucket, dim=1)
            y_ext_one_bucket = T.cat(y_ext_one_bucket, dim=1)
            y_mask_one_bucket = T.cat(y_mask_one_bucket, dim=1)

            y_ext_one_bucket = self.handle_oovs_dynamically(x_ext_one_bucket, y_ext_one_bucket)  # todo

            # y_emb = self.w_rawdata_emb(y)
            # y_shifted = y_emb[:-1, :, :]  # todo : check eos
            y_shifted = ((self.w_rawdata_emb(y_one_bucket))[:-1, :, :]).bfloat16()
            y_shifted = T.cat((Variable(torch.zeros(1, *y_shifted[0].size(), dtype=torch.bfloat16)).to(self.device),
                               y_shifted), 0)

            # h0 = dec_init_state
            if self.cell == "lstm":
                h0_one_bucket = (h0_one_bucket, h0_one_bucket)
            if self.coverage:  # todo : check shape: B * len(x)
                acc_att = Variable(torch.zeros(T.transpose(x_ext_one_bucket, 0, 1).size(),
                                               dtype=torch.bfloat16)).to(self.device)
                # print("acc_att", acc_att.size())

            # print("y_shifted", y_shifted.size())
            # if torch.isnan(y_shifted).sum().item() > 0:  # handles NaN error
            #     print("y_shifted", y_shifted)
            # if torch.isnan(hs_one_bucket).sum().item() > 0:  # handles NaN error
            #     print("hs_one_bucket", hs_one_bucket)
            # if torch.isnan(acc_att).sum().item() > 0:  # handles NaN error
            #     print("acc_att", acc_att)

            with torch.cuda.amp.autocast():
                if self.copy and self.coverage:
                    cost_one_bucket, cost_c \
                        = decoder_low_precision(y_shifted, hs_one_bucket, h0_one_bucket, mask_x_one_bucket,
                                                y_mask_one_bucket,  x_ext_one_bucket, acc_att, max_ext_len,
                                                y_one_bucket, y_ext_one_bucket, self.avg_nll, precision=torch.bfloat16)
                    cost_one_bucket = cost_one_bucket + cost_c
                elif self.copy:  # TODO: refactor  att_dist ?
                    cost_one_bucket \
                        = decoder_low_precision(y_shifted, hs_one_bucket, h0_one_bucket, mask_x_one_bucket,
                                                y_mask_one_bucket, xid=x_ext_one_bucket, max_ext_len=max_ext_len,
                                                y_idx=y_one_bucket, y_ext_idx=y_ext_one_bucket, use_avg_nll=self.avg_nll,
                                                precision=torch.bfloat16)
                elif self.coverage:
                    cost_one_bucket, cost_c \
                        = decoder_low_precision(y_shifted, hs_one_bucket, h0_one_bucket, mask_x_one_bucket,
                                                y_mask_one_bucket, init_coverage=acc_att, y_idx=y_one_bucket,
                                                y_ext_idx=y_ext_one_bucket, use_avg_nll=self.avg_nll,
                                                precision=torch.bfloat16)
                    cost_one_bucket = cost_one_bucket + cost_c
                else:
                    cost_one_bucket \
                        = decoder_low_precision(y_shifted, hs_one_bucket, h0_one_bucket, mask_x_one_bucket,
                                                y_mask_one_bucket, y_idx=y_one_bucket, y_ext_idx=y_ext_one_bucket,
                                                use_avg_nll=self.avg_nll, precision=torch.bfloat16)

            # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
            # cost_without_chain = cost_one_bucket.clone().detach()

            # todo bucket split
            # print("cost_one_bucket", cost_one_bucket.size())
            # print("cost_one_bucket 1", cost_one_bucket)
            split_idx = 0
            for idx_doc in one_bucket:
                # print("x_sections_num[idx_doc]", x_sections_num[idx_doc])
                # print("y_sents_nums[idx_doc]", y_sents_nums[idx_doc])
                cost = cost_one_bucket[split_idx: split_idx + x_sections_num[idx_doc] * y_sents_nums[idx_doc]]
                cost = torch.transpose(cost.view(y_sents_nums[idx_doc], x_sections_num[idx_doc]),
                                       0, 1)  # todo check sequence
                # print("cost 2", cost.size())
                # print("cost 2", cost)
                cost_matrix_list.append(cost.clone())
                split_idx = split_idx + x_sections_num[idx_doc] * y_sents_nums[idx_doc]
            assert split_idx == cost_one_bucket.size(0)

            del cost_one_bucket  # 4 save memory
            del mask_x_one_bucket
            del x_ext_one_bucket
            del hs_one_bucket
            del h0_one_bucket
            del y_one_bucket
            del y_ext_one_bucket
            del y_mask_one_bucket
            del y_shifted
            del acc_att
            # torch.cuda.empty_cache()

            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        assert src_start_idx == hs_all.size(1)
        assert len(cost_matrix_list) == len(y_sents_nums)

        del hs_all
        del h0_all
        # del decoder_low_precision
        # print("cost_matrix_list", cost_matrix_list)
        return cost_matrix_list

    def forward(self, one_batch, matcher=None):
        torch.set_printoptions(profile="full")

        # hs, dec_init_state = self.encode(x, len_x, mask_x)
        # concatenated: x, len_x, mask_x
        # with torch.cuda.amp.autocast(enabled=False):

        concat_x = torch.LongTensor(np.concatenate(one_batch.x_one_batch, axis=1)).to(self.device)
        # concat_len_x = torch.LongTensor([x for len_x in one_batch.len_x_one_batch for x in len_x]).to(self.device)
        concat_len_x = torch.LongTensor([x for len_x in one_batch.len_x_one_batch for x in len_x])
        concat_mask_x = torch.FloatTensor(np.concatenate(one_batch.x_mask_one_batch, axis=1)).to(self.device)

        # if torch.isnan(concat_x).sum().item() > 0:  # handles NaN error
        #     print("concat_x contains Nan.")
        #     concat_x = torch.where(torch.isnan(concat_x), torch.zeros_like(concat_x), concat_x)
        # if torch.isnan(concat_len_x).sum().item() > 0:  # handles NaN error
        #     print("concat_len_x contains Nan.")
        #     concat_len_x = torch.where(torch.isnan(concat_len_x), torch.zeros_like(concat_len_x), concat_len_x)
        # if torch.isnan(concat_mask_x).sum().item() > 0:  # handles NaN error
        #     print("concat_mask_x contains Nan.")
        #     concat_mask_x = torch.where(torch.isnan(concat_mask_x), torch.zeros_like(concat_mask_x), concat_mask_x)

        # print("one_batch.x_one_batch", one_batch.x_one_batch)
        # print("concat_x", concat_x.size())
        # print("concat_x", concat_x)

        # print("one_batch.len_x_one_batch", one_batch.len_x_one_batch)
        # print("concat_len_x", concat_len_x.size())
        # print("concat_len_x", concat_len_x)

        # print("one_batch.x_mask_one_batch", one_batch.x_mask_one_batch)
        # print("concat_mask_x", concat_mask_x.size())
        # print("concat_mask_x", concat_mask_x)
        encodetime = time.time()
        hs, dec_init_state = self.encode(concat_x, concat_len_x, concat_mask_x)
        # print("encode time:", time.time() - encodetime)

        # del concat_x
        # del concat_len_x
        # del concat_mask_x
        # torch.cuda.empty_cache()

        # print("hs", hs.size())
        # print("hs", hs)
        # print("dec_init_state", dec_init_state.size())
        # print("dec_init_state", dec_init_state)

        # use loss as cost todo check
        cost_matrix_listtime = time.time()
        # with self.decoder.half():
        # with torch.no_grad():
        # with torch.cuda.amp.autocast():
        cost_matrix_list = self.construct_cost_with_loss_bucket_version(
            one_batch.x_sections_num, one_batch.y_sents_num,
            one_batch.x_mask_one_batch, one_batch.x_ext_one_batch,
            hs.clone().detach().bfloat16(), dec_init_state.clone().detach().bfloat16(),
            one_batch.y_one_batch, one_batch.y_ext_one_batch, one_batch.y_mask_one_batch,
            one_batch.max_ext_len)

        cost_matrix_tensor = pad_matrix_list_2_tensor(cost_matrix_list, one_batch.x_sections_num,
                                                      one_batch.y_sents_num, self.device)
        # print("cost_matrix_list time:", time.time() - cost_matrix_listtime)

        cost_matrix_tensor = handle_exception_4_cost_matrices(cost_matrix_tensor)
        # del cost_matrix_list

        # print("one_batch.x_sents_num", one_batch.x_sents_num)
        # print("one_batch.y_sents_num", one_batch.y_sents_num)

        article_sections_mask = len_mask(one_batch.x_sections_num, self.device)
        abstract_sents_mask = len_mask(one_batch.y_sents_num, self.device)
        # print("article_sections_mask", article_sections_mask)
        # print("abstract_sents_mask", abstract_sents_mask)

        encoded_sections = reshape_src_articles(dec_init_state, one_batch.x_sections_num, self.device)
        encoded_sections_duplicate = encoded_sections.clone().detach()
        # print("encoded_sections", encoded_sections.size())
        # print("encoded_sections", encoded_sections)
        matchtime = time.time()
        masked_scaling_factors = matcher.predict_alignment_score_4_sections(
            encoded_sections, one_batch.x_sections_num, article_sections_mask)
        # print("match time:", time.time() - matchtime)

        # print("self._epsilon", self._epsilon)
        # print("self._tau_sinkhorn", self._tau_sinkhorn)

        # print("cost_matrix_tensor", cost_matrix_tensor.size())
        # print("cost_matrix_tensor", cost_matrix_tensor)
        # print("masked_scaling_factors 2", masked_scaling_factors.size())
        # print("masked_scaling_factors 2", masked_scaling_factors)
        # print("article_sents_mask", article_sents_mask.size())
        # print("article_sents_mask", article_sents_mask)
        # print("abstract_sents_mask", abstract_sents_mask.size())
        # print("abstract_sents_mask", abstract_sents_mask)

        ottime = time.time()
        # compute mass variations / scaling factors / score functions
        plan_positive, _, src_marginal \
            = compute_sinkhorn_loss(cost_matrix_tensor, masked_scaling_factors,
                                    src_mask=article_sections_mask, tgt_mask=abstract_sents_mask,
                                    epsilon=self._epsilon, tau=self._tau_sinkhorn,
                                    block_padding=3.0)
        # print("ot time:", time.time() - ottime)

        # print("plan_positive", plan_positive.size())
        # print("plan_positive", plan_positive)
        # print("hs1", hs)
        # print("dec_init_state1", dec_init_state)

        # print("++++++++++++++++")
        decodetime = time.time()
        # with torch.cuda.amp.autocast(enabled=False):
        filtered_cost, filtered_cost_c = self.filter_and_recombine_s2s(
            plan_positive, one_batch.x_sections_num, one_batch.y_sents_num,
            one_batch.x_mask_one_batch, one_batch.x_ext_one_batch,
            hs, dec_init_state,
            one_batch.y_one_batch, one_batch.y_ext_one_batch, one_batch.y_mask_one_batch, one_batch.len_y_one_batch,
            one_batch.max_ext_len)
        # print("decode time:", time.time() - decodetime)
        # print("++++++++++++++++")

        # print("hs2", hs)
        # print("dec_init_state2", dec_init_state)

        # print("weighted_filtered_cost", weighted_filtered_cost.size())
        # print("weighted_filtered_cost_c", weighted_filtered_cost_c.size())
        #
        # print("src_marginal", src_marginal.size())
        # print("src_marginal", src_marginal)
        # print("masked_scaling_factors", masked_scaling_factors.size())
        # print("masked_scaling_factors", masked_scaling_factors)

        # kl_div_cost = torch.sum(generalized_kl_div(src_marginal, masked_scaling_factors), dim=1) / \
        #     (torch.FloatTensor(one_batch.x_sections_num).to(self.device))
        # print("torch.FloatTensor(one_batch.x_sents_num)", torch.FloatTensor(one_batch.x_sents_num))
        # kl_div_cost = torch.mean(kl_div_cost)
        # print("kl_div_cost", kl_div_cost.size())
        # print("kl_div_cost", kl_div_cost)

        # print("ot_cost_positive", ot_cost_positive)
        # ot_cost_positive = ot_cost_positive / (torch.FloatTensor(one_batch.y_sents_num).to(self.device))
        # ot_cost_positive = torch.mean(ot_cost_positive)
        # print("ot_cost_positive", ot_cost_positive.size())
        # print("ot_cost_positive", ot_cost_positive)

        # cost_matcher = ot_cost_positive + kl_div_cost * self._tau_nn
        # print("self._tau_nn", self._tau_nn)
        # print("cost_matcher", cost_matcher)

        src_marginal_duplicate = src_marginal.clone().detach()

        if self.coverage:
            return (encoded_sections_duplicate, src_marginal_duplicate), \
                   filtered_cost, filtered_cost_c
        else:
            return (encoded_sections_duplicate, src_marginal_duplicate), \
                   filtered_cost, None


'''
   def handle_oovs_dynamically(self, x_ext_tensor, y_ext_tensor):
        x_ext_tensor = torch.transpose(x_ext_tensor, 0, 1)
        y_ext_tensor = torch.transpose(y_ext_tensor, 0, 1)

        print("x_ext_tensor", x_ext_tensor.size())
        print("y_ext_tensor", y_ext_tensor.size())
        print("self.dict_size", self.dict_size)
        # print(,"")
        select_mask = torch.ge(y_ext_tensor, self.dict_size)  # get index
        print("select_mask", select_mask.size())
        # print("select_mask", select_mask)

        oov_idx_in_dict = torch.masked_select(y_ext_tensor, select_mask)  # src 1st, tgt 2nd
        print("oov_idx_in_dict", len(oov_idx_in_dict))
        print("oov_idx_in_dict", oov_idx_in_dict)

        pair_idx = torch.nonzero(select_mask)
        print("pair_idx 0 ", pair_idx.size())
        print("pair_idx 0 ", pair_idx)

        print("pair_idx.size(0)", pair_idx.size(0))

        for i in range(pair_idx.size(0)):
            print("pair_idx[i, 0]", pair_idx[i, 0])
            print("pair_idx[i, 1]", pair_idx[i, 1])
            print("y_ext_tensor[pair_idx[i, 0], pair_idx[i, 1]]", y_ext_tensor[pair_idx[i, 0], pair_idx[i, 1]])
            print("x_ext_tensor[pair_idx[i, 0]]", x_ext_tensor[pair_idx[i, 0]].size())
            print("x_ext_tensor[pair_idx[i, 0]]", x_ext_tensor[pair_idx[i, 0]])
            print("oov_idx_in_dict[i]", oov_idx_in_dict[i])
            assert y_ext_tensor[pair_idx[i, 0], pair_idx[i, 1]] == oov_idx_in_dict[i]

            if y_ext_tensor[pair_idx[i, 0], pair_idx[i, 1]] not in x_ext_tensor[pair_idx[i, 0]]:
                y_ext_tensor[pair_idx[i, 0], pair_idx[i, 1]] = self.lfw_idx
                print("y_ext_tensor[pair_idx[i, 0], pair_idx[i, 1]]", y_ext_tensor[pair_idx[i, 0], pair_idx[i, 1]])

            print("-------------------------------------------------------")

        # self.lfw_idx
        return torch.transpose(y_ext_tensor, 0, 1)
        
def compute_mass_variation(self):  # 4 ot

    return
        
        # expand: reuse concat_mask_x to save memory
        # concat_mask_x, x_ext, hs, h0, y, y_ext, mask_y =  \
        #     self.data_expand(one_batch.sents_num_stats_one_batch,
        #                      concat_mask_x, one_batch.x_ext_one_batch,
        #                      hs, dec_init_state,
        #                      one_batch.y_one_batch, one_batch.y_ext_one_batch, one_batch.y_mask_one_batch
        #                      )
def universal_match(self):  # 4 ot, match each pair of src and abst sentences

    return
        
torch.LongTensor(x).to(options["device"]),
torch.LongTensor(len_x).to(options["device"]), \
torch.LongTensor(y).to(options["device"]),
torch.FloatTensor(x_mask).to(options["device"]), \
torch.FloatTensor(y_mask).to(options["device"]),
torch.LongTensor(x_ext).to(options["device"]), \
torch.LongTensor(y_ext).to(options["device"]), \
batch.max_ext_len,

if need_rouge:
    handle_one_batch_4_early_stage(batch_raw, modules, consts, options, need_rouge=True)
else:
    handle_one_batch_4_early_stage(batch_raw, modules, consts, options, need_rouge=True)

def handle_one_batch_4_early_stage(batch_raw, modules, consts, options, need_rouge=True):  # need rouge
    # return y_pred, cost, cost_c
    return


def handle_one_batch_4_later_stage():  # not need rouge

    return
    
print("x_list", len(x_list))
print("x_list", x_list[0].shape)
print("x_list", np.concatenate(x_list, axis=1).shape)

print("len_x_list", len(len_x_list))
print("len_x_list", len_x_list[0])
print("len_x_list", [x for len_x in len_x_list for x in len_x])

print("mask_x_list", len(mask_x_list))
print("mask_x_list", mask_x_list[0].shape)
print("mask_x_list", np.concatenate(mask_x_list, axis=1).shape)

for idx in range(len(x_list)):  # batch_size = len(x_list)
    # todo: ?? sort & sort back
    torch.LongTensor(x).to(options["device"])
    torch.LongTensor(len_x).to(options["device"])
    torch.FloatTensor(x_mask).to(options["device"])
    x = x_list[idx]

hs, dec_init_state = self.encode(x_list[idx], len_x_list[idx], mask_x_list[idx])

    def encode_multiple_articles(self, x_list, len_x_list, mask_x_list):  # 4 ot
    concat_x = torch.LongTensor(np.concatenate(x_list, axis=1)).to(self.device)
    concat_len_x = torch.LongTensor([x for len_x in len_x_list for x in len_x]).to(self.device)
    concat_mask_x = torch.FloatTensor(np.concatenate(mask_x_list, axis=1)).to(self.device)

    hs, dec_init_state = self.encode(concat_x, concat_len_x, concat_mask_x)
    return hs, dec_init_state
hs, dec_init_state = self.encode_multiple_articles(
    one_batch.x_one_batch, one_batch.len_x_one_batch, one_batch.x_mask_one_batch)
    
        def data_expand(self, sents_nums, concat_mask_x, x_ext_list, hs, h0, y_list, y_ext_list, y_mask_list):
        # src side
        expanded_mask_x = []
        expanded_x_ext = []

        expanded_hs = []
        expanded_h0 = []

        # tgt side
        expanded_y = []
        expanded_y_ext = []
        expanded_y_mask = []

        src_start_idx = 0
        abst_start_idx = 0

        for idx_doc in range(len(sents_nums)):  # sents_nums: {(x_sents_num, y_sents_num)}
            # src side
            mask_x = concat_mask_x[:, src_start_idx:src_start_idx + sents_nums[idx_doc][0], :]
            print("mask_x 1", mask_x.size())
            mask_x = mask_x.repeat(1, sents_nums[idx_doc][1], 1)
            print("mask_x 2", mask_x.size())
            expanded_mask_x.append(mask_x)

            x_ext = torch.LongTensor(x_ext_list[idx_doc]).to(self.device)
            print("x_ext", x_ext.size())
            expanded_x_ext.append(x_ext.repeat(1, sents_nums[idx_doc][1]))

            hs_1_doc = hs[:, src_start_idx:src_start_idx + sents_nums[idx_doc][0], :]
            hs_1_doc = hs_1_doc.repeat(1, sents_nums[idx_doc][1], 1)
            expanded_hs.append(hs_1_doc)

            h0_1_doc = h0[src_start_idx:src_start_idx + sents_nums[idx_doc][0], :]
            h0_1_doc = h0_1_doc.repeat(sents_nums[idx_doc][1], 1)
            expanded_h0.append(h0_1_doc)

            # tgt side
            y = torch.LongTensor(y_list[idx_doc]).to(self.device)
            print("y", y.size())
            expanded_y.append(y.repeat(1, sents_nums[idx_doc][0]))
            print("y", y.size())

            y_ext_1_doc = torch.LongTensor(y_ext_list[idx_doc]).to(self.device)
            print("y_ext_1_doc", y_ext_1_doc.size())
            expanded_y_ext.append(y_ext_1_doc.repeat(1, sents_nums[idx_doc][0]))

            y_mask_1_doc = torch.FloatTensor(y_mask_list[idx_doc]).to(self.device)
            print("y_mask_1_doc", y_mask_1_doc.size())
            expanded_y_mask.append(y_mask_1_doc.repeat(1, sents_nums[idx_doc][0], 1))

            src_start_idx += sents_nums[idx_doc][0]
            abst_start_idx += sents_nums[idx_doc][1]
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        print("src_start_idx", src_start_idx)
        print("abst_start_idx", abst_start_idx)

        expanded_mask_x = T.cat(expanded_mask_x, dim=1)
        expanded_x_ext = T.cat(expanded_x_ext, dim=1)

        expanded_hs = T.cat(expanded_hs, dim=1)
        expanded_h0 = T.cat(expanded_h0, dim=0)

        expanded_y = T.cat(expanded_y, dim=1)
        expanded_y_ext = T.cat(expanded_y_ext, dim=1)
        expanded_y_mask = T.cat(expanded_y_mask, dim=1)

        print("expanded_mask_x", expanded_mask_x.size())
        print("expanded_x_ext", expanded_x_ext.size())

        print("expanded_hs", expanded_hs.size())
        print("expanded_h0", expanded_h0.size())

        print("expanded_y", expanded_y.size())
        print("expanded_y_ext", expanded_y_ext.size())
        print("expanded_y_mask", expanded_y_mask.size())

        return expanded_mask_x, expanded_x_ext, expanded_hs, expanded_h0, \
            expanded_y, expanded_y_ext, expanded_y_mask
'''
