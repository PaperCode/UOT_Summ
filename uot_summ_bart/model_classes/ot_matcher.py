# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict

# from one_sentence_pgnet import *
from .utils_4_uot import *


class OTMatcher(nn.Module):
    def __init__(self, input_dimension=1024, dim_section_ext=256, hidden_size_ot_extractor=128):
        super(OTMatcher, self).__init__()

        self._input_dimension = input_dimension
        self._dim_sec_ext = dim_section_ext
        self._hidden_size_ot = hidden_size_ot_extractor

        # self.is_bidirectional_sec_ext = options["is_bidirectional_sec_ext"]
        self.section_aggregator = nn.Linear(self._input_dimension, self._dim_sec_ext)

        self.is_bidirectional_sec_ext = True
        self.section_score = nn.LSTM(self._dim_sec_ext, self._hidden_size_ot,
                                     bidirectional=self.is_bidirectional_sec_ext)

        # self.device = options["device"]

        # self.one_sen_pgnet = OneSentencePGNET(modules, consts, options)
        self._mass_variation_cal = nn.Sequential(OrderedDict([
            ('mass_fc1',
             nn.Linear(self._hidden_size_ot * 2 if self.is_bidirectional_sec_ext else self._hidden_size_ot, 1)),
            ('relu', nn.ReLU())
            ]))
        self.init_weights()

    def init_weights(self):
        init_linear_weight(self.section_aggregator)

        init_lstm_weight(self.section_score)

        # init.xavier_normal_(self._mass_variation_cal.mass_fc1.weight)
        init.kaiming_normal_(self._mass_variation_cal.mass_fc1.weight, mode='fan_in', nonlinearity='relu')

        init.constant_(self._mass_variation_cal.mass_fc1.bias, 0.)

    def aggregate_section_masked(self, emb_x, mask_x):
        # sec_num * sec_length * embedding_size
        mask_x = mask_x.float().unsqueeze(2)
        # print("emb_x", emb_x.size())
        # print("mask_x", mask_x.size())
        # print("mask_x", mask_x)
        # print("torch.sum(emb_x * mask_x, 1)", torch.sum(emb_x * mask_x, 1).size())
        # print("torch.sum(mask_x, dim=1)", torch.sum(mask_x, dim=1))
        aggregated_sections = torch.sum(emb_x * mask_x, 1) / torch.sum(mask_x, dim=1)
        # print("aggregated_sections 1", aggregated_sections.size())

        aggregated_sections = torch.tanh(self.section_aggregator(aggregated_sections))
        # print("aggregated_sections 2", aggregated_sections.size())

        return aggregated_sections

    # def predict_alignment_score_4_sections(self, emb_x, len_x, mask_x):
    def predict_alignment_score_4_sections(self, emb_x, mask_x):
        # print("emb_x", emb_x.size())
        # print("emb_x", emb_x)
        # print("mask_x", mask_x.size())
        # print("mask_x", mask_x)

        aggregated_sections = self.aggregate_section_masked(emb_x, mask_x.float())
        # print("len_x", len_x)

        self.section_score.flatten_parameters()
        # emb_x = torch.nn.utils.rnn.pack_padded_sequence(emb_x, len_x, enforce_sorted=False)  # todo 4 ot
        # emb_x = torch.nn.utils.rnn.pack_padded_sequence(
        #     aggregated_sections.unsqueeze(1),
        #     [1], batch_first=False, enforce_sorted=False)  # todo  enforce_sorted?

        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        # hs, _ = self.section_score(aggregated_sections, None)
        hs, _ = self.section_score(aggregated_sections.unsqueeze(1), None)
        # hs, _ = torch.nn.utils.rnn.pad_packed_sequence(hs)
        # print("hs", hs.size())

        scaling_factors = self._mass_variation_cal(hs)
        # print("scaling_factors", scaling_factors.size())
        # print("scaling_factors", scaling_factors)

        return torch.squeeze(torch.squeeze(scaling_factors, dim=2), dim=1)
        # return torch.squeeze(scaling_factors, dim=2).transpose(0, 1)

    def forward(self, encoded_sections, mask_x, labels):
        # print("encoded_sections matcher", encoded_sections.size())
        # print("encoded_sections 2", encoded_sections)
        # print("labels 0", labels.size(0))
        # print("labels", labels.size())
        # print("labels", labels)
        # print("mask_x", mask_x)

        # article_sections_mask = len_mask(x_sections_num, encoded_sections.device)

        masked_section_scores = self.predict_alignment_score_4_sections(encoded_sections, mask_x)
        # print("masked_section_scores", masked_section_scores.size())
        # print("masked_section_scores", masked_section_scores)

        # print("torch.FloatTensor(one_batch.x_sections_num).to(self.device)",
        #       torch.FloatTensor(one_batch.x_sections_num).to(self.device))

        l2_loss = torch.sum(torch.square(masked_section_scores - labels), dim=0) / labels.size(0)
        # (torch.FloatTensor(x_sections_num).to(encoded_sections.device))

        # print("l2_loss", l2_loss)

        return l2_loss


'''
    def forward(self, data_pack, one_batch, current_device):
        (encoded_sections, labels) = data_pack
        # print("encoded_sections 2", encoded_sections.size())
        # print("encoded_sections 2", encoded_sections)
        # print("labels", labels.size())
        # print("labels", labels)

        article_sections_mask = len_mask(one_batch.x_sections_num, current_device)

        masked_section_scores = self.predict_alignment_score_4_sections(
            encoded_sections, one_batch.x_sections_num, article_sections_mask)

        # print("masked_section_scores", masked_section_scores.size())
        # print("masked_section_scores", masked_section_scores)
        # print("torch.FloatTensor(one_batch.x_sections_num).to(self.device)",
        #       torch.FloatTensor(one_batch.x_sections_num).to(self.device))

        l2_loss = torch.sum(torch.square(masked_section_scores - labels), dim=1) / \
            (torch.FloatTensor(one_batch.x_sections_num).to(current_device))

        # print("l2_loss", l2_loss)

        return l2_loss.mean()

'''
