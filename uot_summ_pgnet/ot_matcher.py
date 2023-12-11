# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict

# from one_sentence_pgnet import *
from utils_pg import *


class OTMatcher(nn.Module):
    def __init__(self, modules, consts, options):
        super(OTMatcher, self).__init__()

        self.dim_sec_ext = consts["dim_section_ext"]
        self.hidden_size_ot = consts["hidden_size_ot_extractor"]
        self.is_bidirectional_sec_ext = options["is_bidirectional_sec_ext"]
        self.section_score = nn.LSTM(self.dim_sec_ext, self.hidden_size_ot,
                                     bidirectional=self.is_bidirectional_sec_ext)

        self.device = options["device"]

        # self.one_sen_pgnet = OneSentencePGNET(modules, consts, options)
        self._mass_variation_cal = nn.Sequential(OrderedDict([
            ('mass_fc1',
             nn.Linear(self.hidden_size_ot * 2 if self.is_bidirectional_sec_ext else self.hidden_size_ot, 1)),
            ('relu', nn.ReLU())
            ]))
        self.init_weights()

    def init_weights(self):
        init_lstm_weight(self.section_score)

        # init.xavier_normal_(self._mass_variation_cal.mass_fc1.weight)
        init.kaiming_normal_(self._mass_variation_cal.mass_fc1.weight, mode='fan_in', nonlinearity='relu')

        init.constant_(self._mass_variation_cal.mass_fc1.bias, 0.)

    def predict_alignment_score_4_sections(self, emb_x, len_x, mask_x):
        # print("emb_x", emb_x.size())
        # print("len_x", len_x)
        # print("mask_x", mask_x.size())
        # print("mask_x", mask_x)

        self.section_score.flatten_parameters()
        emb_x = torch.nn.utils.rnn.pack_padded_sequence(torch.transpose(emb_x, 0, 1),
                                                        len_x, enforce_sorted=False)  # todo  enforce_sorted?

        hs, _ = self.section_score(emb_x, None)  # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        hs, _ = torch.nn.utils.rnn.pad_packed_sequence(hs)
        # print("hs", hs.size())

        scaling_factors = self._mass_variation_cal(hs)
        # print("scaling_factors", scaling_factors.size())
        # print("scaling_factors", scaling_factors)

        return torch.squeeze(scaling_factors, dim=2).transpose(0, 1) * mask_x

    def forward(self, data_pack, one_batch):
        (encoded_sections, labels) = data_pack
        # print("encoded_sections 2", encoded_sections.size())
        # print("encoded_sections 2", encoded_sections)
        # print("labels", labels.size())
        # print("labels", labels)

        article_sections_mask = len_mask(one_batch.x_sections_num, self.device)

        masked_section_scores = self.predict_alignment_score_4_sections(
            encoded_sections, one_batch.x_sections_num, article_sections_mask)

        # print("masked_section_scores", masked_section_scores.size())
        # print("masked_section_scores", masked_section_scores)
        # print("torch.FloatTensor(one_batch.x_sections_num).to(self.device)",
        #       torch.FloatTensor(one_batch.x_sections_num).to(self.device))

        l2_loss = torch.sum(torch.square(masked_section_scores - labels), dim=1) / \
            (torch.FloatTensor(one_batch.x_sections_num).to(self.device))

        # print("l2_loss", l2_loss)

        return l2_loss.mean()

