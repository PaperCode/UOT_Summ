# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import numpy as np
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable

from utils_pg import *
# from gru_dec import *
# from lstm_dec_v2 import *
# from lstm_dec_memory_optimized import *
from lstm_dec_4_ot import *

from word_prob_layer import *


class Model4OT(nn.Module):
    def __init__(self, modules, consts, options):
        super(Model4OT, self).__init__()
        ##########################################
        self.pattern_type = options["pattern"]

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
        self.len_x = consts["len_x"]
        self.len_y = consts["len_y"]
        self.hidden_size = consts["hidden_size"]
        self.dict_size = consts["dict_size"] 
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
        self.encoder.flatten_parameters()
        emb_x = self.w_rawdata_emb(x)
        # print("emb_x1", emb_x.data.size())
        # print("emb_x1", emb_x)
        # emb_x = torch.nn.utils.rnn.pack_padded_sequence(emb_x, len_x)
        emb_x = torch.nn.utils.rnn.pack_padded_sequence(emb_x, len_x, enforce_sorted=False)  # todo 4 ot

        # print("emb_x", emb_x.data.size())
        # print("emb_x", emb_x)

        hs, hn = self.encoder(emb_x, None)
        hs, _ = torch.nn.utils.rnn.pad_packed_sequence(hs)
         
        dec_init_state = T.sum(hs * mask_x, 0) / T.sum(mask_x, 0)
        dec_init_state = T.tanh(self.get_dec_init_state(dec_init_state))
        return hs, dec_init_state

    def compute_mass_variation(self):  # 4 ot

        return

    def universal_match(self):  # 4 ot, match each pair of src and abst sentences

        return

    def decode_once(self, y, hs, dec_init_state, mask_x, x=None, max_ext_len=None, acc_att=None):
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

    def matrix_recover(self, ):

        return

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

    def forward_4_early_epochs(self, one_batch, matcher=None):
        # hs, dec_init_state = self.encode(x, len_x, mask_x)
        # concatenated: x, len_x, mask_x
        concat_x = torch.LongTensor(np.concatenate(one_batch.x_one_batch, axis=1)).to(self.device)
        concat_len_x = torch.LongTensor([x for len_x in one_batch.len_x_one_batch for x in len_x]).to(self.device)
        concat_mask_x = torch.FloatTensor(np.concatenate(one_batch.x_mask_one_batch, axis=1)).to(self.device)

        hs, dec_init_state = self.encode(concat_x, concat_len_x, concat_mask_x)

        print("concat_x", concat_x.size())
        print("concat_mask_x", concat_mask_x.size())
        print("hs", hs.size())
        print("dec_init_state", dec_init_state.size())

        # expand: reuse concat_mask_x to save memory
        concat_mask_x, x_ext, hs, h0, y, y_ext, mask_y =  \
            self.data_expand(one_batch.sents_num_stats_one_batch,
                             concat_mask_x, one_batch.x_ext_one_batch,
                             hs, dec_init_state,
                             one_batch.y_one_batch, one_batch.y_ext_one_batch, one_batch.y_mask_one_batch
                             )

        # y_emb = self.w_rawdata_emb(y)
        # y_shifted = y_emb[:-1, :, :]  # todo : check eos
        y_shifted = (self.w_rawdata_emb(y))[:-1, :, :]
        y_shifted = T.cat((Variable(torch.zeros(1, *y_shifted[0].size())).to(self.device), y_shifted), 0)

        # h0 = dec_init_state
        if self.cell == "lstm":
            h0 = (h0, h0)
        if self.coverage:  # todo : check shape: B * len(x)
            acc_att = Variable(torch.zeros(T.transpose(x_ext, 0, 1).size())).to(self.device)

        print("y_shifted", y_shifted.size())

        if self.copy and self.coverage:
            y_pred, cost, cost_c = self.decoder(y_shifted, hs, h0, concat_mask_x, mask_y, \
                    x_ext, acc_att, one_batch.max_ext_len, y, y_ext, self.avg_nll)
        elif self.copy:  # TODO: refactor  att_dist ?
            y_pred, cost = self.decoder(y_shifted, hs, h0, concat_mask_x, mask_y, \
                    xid=x_ext, max_ext_len=one_batch.max_ext_len, y_idx=y, y_ext_idx=y_ext, use_avg_nll=self.avg_nll)
        elif self.coverage:
            y_pred, cost, cost_c = self.decoder(y_shifted, hs, h0, concat_mask_x, mask_y, \
                    init_coverage=acc_att, y_idx=y, y_ext_idx=y_ext, use_avg_nll=self.avg_nll)
        else:
            y_pred, cost = self.decoder(y_shifted, hs, h0, concat_mask_x, mask_y, \
                    y_idx=y, y_ext_idx=y_ext, use_avg_nll=self.avg_nll)

        # compute_mass_variation

        if self.coverage:
            return y_pred, cost, cost_c
        else:
            return y_pred, cost, None

    def forward_4_later_epochs(self, one_batch, matcher=None):
        # compute_mass_variation
        self.universal_match()

        return

    def forward(self, one_batch_data, is_early=True, matcher=None):
        if is_early:  # use ROUGE
            return self.forward_4_early_epochs(one_batch_data, matcher=matcher)
        else:
            return self.forward_4_later_epochs(one_batch_data, matcher=matcher)


'''
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
'''
