### memory optimized version of lstm v2
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from word_prob_layer_in_decoder import *  # new

import math
import time

from utils_pg import *


class LSTM4OTDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, ctx_size, device, copy, coverage, is_predicting, dict_size):
        super(LSTM4OTDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        # TODO: in the latter version, this should be removed to the forward parameter
        self.is_predicting = is_predicting
        self.device = device
        self.copy = copy
        self.coverage = coverage

        self.lstm_1 = nn.LSTMCell(self.input_size, self.hidden_size)

        self.Wx = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.ctx_size))
        self.Ux = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.hidden_size))
        self.bx = nn.Parameter(torch.Tensor(4 * self.hidden_size))

        self.Wc_att = nn.Parameter(torch.Tensor(self.ctx_size, self.ctx_size))
        self.b_att = nn.Parameter(torch.Tensor(self.ctx_size))

        self.W_comb_att = nn.Parameter(torch.Tensor(self.ctx_size, 2 * self.hidden_size))
        self.U_att = nn.Parameter(torch.Tensor(1, self.ctx_size))

        if self.coverage:
            self.W_coverage = nn.Parameter(torch.Tensor(self.ctx_size, 1))

        ##########################################
        self.dict_size = dict_size
        self.word_prob = WordProbLayerInDecoder(self.hidden_size, self.ctx_size, self.input_size, \
                                                self.dict_size, self.device, self.copy, self.coverage)
        ##########################################

        self.init_weights()

    def init_weights(self):
        init_lstm_weight(self.lstm_1)
        init_ortho_weight(self.Wx)
        init_ortho_weight(self.Ux)
        init_bias(self.bx)
        init_ortho_weight(self.Wc_att)
        init_bias(self.b_att)
        init_ortho_weight(self.W_comb_att)
        init_ortho_weight(self.U_att)
        if self.coverage:
            init_ortho_weight(self.W_coverage)

    #################################
    @staticmethod
    def one_step_nll_loss(y_pred, y):
        # print("y_pred", y_pred)
        # print("y_pred", y_pred.size())
        cost = -T.log(T.gather(y_pred, 1, y.view(y.size(0), 1)))  # the content of y/y_ext is useful here!
        # if torch.isinf(cost).sum().item() > 0:  # handles Inf error
        #     # cost = torch.where(torch.isinf(cost), torch.ones_like(cost) * 20.0, cost)
        #     print("costaaa", cost.size())
        #     print("cost aaa", cost)
        #     print("y", y)
        #     print("y", y.size())
        #     print("sssssssssssssssssssssssssssssssssssssss")

        return cost.view(y.shape)

    @staticmethod
    def nll_loss_in_all(cost, y, y_mask, avg=True):
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = T.sum(cost * y_mask, 0) / T.sum(y_mask, 0)
        else:
            cost = T.sum(cost * y_mask, 0)

        # cost = cost.view((y.size(1), -1))
        cost = cost.view((y.size(1), -1)).squeeze()
        # print("cost sss", cost.size())
        # print("cost sss", cost)

        return cost
        # return T.mean(cost)

    def get_word_atten(self, pctx, h1, x_mask, acc_att=None):  # acc_att: B * len(x)
        if acc_att is not None:
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print("h1", h1.size())
            # print("self.W_comb_at", self.W_comb_att.size())
            # print("acc_att", acc_att.size())
            # print("self.W_coverage", self.W_coverage.size())
            # print("F.linear(h1, self.W_comb_att)", F.linear(h1, self.W_comb_att).size())
            # print("F.linear(T.transpose(acc_att, 0, 1).unsqueeze(2), self.W_coverage)",
            #       F.linear(T.transpose(acc_att, 0, 1).unsqueeze(2), self.W_coverage).size())

            h = F.linear(h1, self.W_comb_att) \
                + F.linear(T.transpose(acc_att, 0, 1).unsqueeze(2), self.W_coverage)  # len(x) * B * ?
        else:
            h = F.linear(h1, self.W_comb_att)
        # print("pctx", pctx.size())
        # print("x_mask", x_mask.size())
        unreg_att = T.tanh(pctx + h) * x_mask
        unreg_att = F.linear(unreg_att, self.U_att)

        word_atten = T.exp(unreg_att - T.max(unreg_att, 0, keepdim=True)[0]) * x_mask
        sum_word_atten = T.sum(word_atten, 0, keepdim=True)
        word_atten = word_atten / sum_word_atten
        return word_atten

    def recurrence(self, x, y_mask, hidden, pctx, context, x_mask, acc_att=None):
        # tttt1 = time.time()
        pre_h, pre_c = hidden

        h1, c1 = self.lstm_1(x, hidden)
        h1 = y_mask * h1 + (1.0 - y_mask) * pre_h
        c1 = y_mask * c1 + (1.0 - y_mask) * pre_c

        # tttt2 = time.time()

        # len(x) * batch_size * 1
        s = T.cat((h1.view(-1, self.hidden_size), c1.view(-1, self.hidden_size)), 1)
        # tttt3 = time.time()

        if self.coverage:
            word_atten = self.get_word_atten(pctx, s, x_mask, acc_att)
        else:
            word_atten = self.get_word_atten(pctx, s, x_mask)
        # tttt4 = time.time()

        atted_ctx = T.sum(word_atten * context, 0)

        # tttt5 = time.time()

        ifoc_preact = F.linear(h1, self.Ux) + F.linear(atted_ctx, self.Wx, self.bx)
        x4i, x4f, x4o, x4c = ifoc_preact.chunk(4, 1)

        # tttt6 = time.time()

        i = torch.sigmoid(x4i)
        f = torch.sigmoid(x4f)
        o = torch.sigmoid(x4o)
        # tttt7 = time.time()
        c2 = f * c1 + i * torch.tanh(x4c)
        h2 = o * torch.tanh(c2)
        c2 = y_mask * c2 + (1.0 - y_mask) * c1
        h2 = y_mask * h2 + (1.0 - y_mask) * h1
        # tttt8 = time.time()
        word_atten_ = T.transpose(word_atten.view(x_mask.size(0), -1), 0, 1)

        # tttt9 = time.time()
        # if tttt9 - tttt1 > 0.2:  # 123567
        #     print(tttt2 - tttt1, "/ ", tttt3 - tttt2, "/ ", tttt4 - tttt3, "/ ", tttt5 - tttt4, "/ ",
        #           tttt6 - tttt5, "/ ", tttt7 - tttt6, "/ ",  tttt8 - tttt7, "/ ", tttt9 - tttt8, "/ ")

        if self.coverage:
            acc_att += word_atten_
            return (h2, c2), h2, atted_ctx, word_atten_, acc_att
        else:
            return (h2, c2), h2, atted_ctx, word_atten_

    # def forward(self, y_emb, context, init_state, x_mask, y_mask, xid=None, init_coverage=None): #original interface
    def forward(self, y_emb, context, init_state, x_mask, y_mask, xid=None, init_coverage=None, \
                max_ext_len=None, y_idx=None, y_ext_idx=None, use_avg_nll=True, precision=torch.float):
        if self.is_predicting:  # TODO: to modify-> not based on training and testing,but forward once, or consecutively
            hidden = init_state
            acc_att = init_coverage
            if self.copy:
                xid = T.transpose(xid, 0, 1)  # B * len(x)

            pctx = F.linear(context, self.Wc_att, self.b_att)

            if self.coverage:
                hidden, s, att, att_dist, acc_att = self.recurrence(y_emb, y_mask, hidden, pctx, context, x_mask, acc_att)
            else:
                hidden, s, att, att_dist = self.recurrence(y_emb, y_mask, hidden, pctx, context, x_mask)

            if self.copy:
                y_pred_once = self.word_prob(s, att, y_emb, att_dist, xid, max_ext_len)
            else:
                y_pred_once = self.word_prob(s, att, y_emb)

            if self.coverage:
                return y_pred_once, (hidden[0], hidden[1]), acc_att
            else:
                return y_pred_once, (hidden[0], hidden[1])
        else:  # training
            dists, Cs, cost_total = [], [], []
            hidden = init_state
            acc_att = init_coverage
            if self.copy:
                xid = T.transpose(xid, 0, 1)  # B * len(x)

            # print("context", context)
            # print("self.Wc_att", self.Wc_att)
            # print("self.b_att", self.b_att)

            pctx = F.linear(context, self.Wc_att, self.b_att)
            # x = y_emb
            # print("y_emb", y_emb.size())

            # if torch.isnan(pctx).sum().item() > 0:  # handles NaN error
            #     print("pctx", pctx)

            # decode_step_time = time.time()
            # one_step_time = ""
            steps = range(y_emb.size(0))
            for i in steps:
                # time_per_step = time.time()

                if self.coverage:
                    Cs += [acc_att]
                    hidden, s, att, att_dist, acc_att = self.recurrence(y_emb[i], y_mask[i], hidden, pctx,
                                                                        context, x_mask, acc_att)
                else:
                    hidden, s, att, att_dist = self.recurrence(y_emb[i], y_mask[i], hidden, pctx, context, x_mask)

                # one_step_time = one_step_time + str(round(time.time() - time_per_step, 3)) + "* "
                # time_per_step = time.time()
                # if torch.isnan(hidden).sum().item() > 0:  # handles NaN error
                #     print(i, "hidden", hidden)
                # if torch.isnan(s).sum().item() > 0:  # handles NaN error
                #     print(i, "s", s)
                # if torch.isnan(att).sum().item() > 0:  # handles NaN error
                #     print(i, "att", att)
                # if torch.isnan(att_dist).sum().item() > 0:  # handles NaN error
                #     print(i, "att_dist", att_dist)
                # if torch.isnan(acc_att).sum().item() > 0:  # handles NaN error
                #     print(i, "acc_att", acc_att)
                ##########################################
                if self.copy:
                    y_pred_once = self.word_prob(s, att, y_emb[i], att_dist, xid, max_ext_len, preci=precision)
                    cost_once = self.one_step_nll_loss(y_pred_once, y_ext_idx[i])
                else:
                    y_pred_once = self.word_prob(s, att, y_emb[i], preci=precision)
                    cost_once = self.one_step_nll_loss(y_pred_once, y_idx[i])
                cost_total += [cost_once]
                ##########################################
                dists += [att_dist]
                # one_step_time = one_step_time + str(round(time.time() - time_per_step, 3)) + "// "

            # print("step time details: ", one_step_time)
            # print("decode step time ", time.time() - decode_step_time)
            # print("decode steps ", steps)
            # print("Cs", Cs)

            with torch.cuda.amp.autocast(enabled=False):
                if self.coverage:
                    Cs = T.stack(Cs).view(y_emb.size(0), *Cs[0].size())
                cost_total = T.stack(cost_total).view(y_emb.size(0), *cost_total[0].size())

            if self.copy:
                cost_per_batch = self.nll_loss_in_all(cost_total, y_ext_idx, y_mask, use_avg_nll)
            else:
                cost_per_batch = self.nll_loss_in_all(cost_total, y_idx, y_mask, use_avg_nll)

            with torch.cuda.amp.autocast(enabled=False):
                dists = T.stack(dists).view(y_emb.size(0), *dists[0].size())

            if self.coverage:
                # print("y_mask", y_mask.size())

                # print("T.sum(T.min(dists, Cs)",  T.min(dists, Cs).size())
                # print("T.sum(T.min(dists, Cs), 2)",  T.sum(T.min(dists, Cs), 2).size())
                # old:
                # cost_coverage = T.mean(T.sum(T.min(dists, Cs), 2))
                # print("cost_coverage", cost_coverage)

                cost_coverage = T.mean(T.sum(T.min(dists, Cs), 2), dim=0)
                # print("cost_coverage", cost_coverage.size())
                # cost_coverage = T.mean(cost_coverage)
                return cost_per_batch, cost_coverage
            else:
                return cost_per_batch

