#!/usr/bin/env python
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Tuple, Dict, List
from models.modules.context_embedder_base import ContextEmbedderBase
from models.few_shot_text_classifier import FewShotTextClassifier
from models.few_shot_seq_labeler import FewShotSeqLabeler
from collections import Counter


class FewShotSLU(torch.nn.Module):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: FewShotTextClassifier,
                 slot_decoder: FewShotSeqLabeler,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None):
        super(FewShotSLU, self).__init__()
        self.opt = opt
        self.context_embedder = context_embedder
        self.intent_decoder = intent_decoder
        self.slot_decoder = slot_decoder
        self.config = config
        self.emb_log = emb_log

        self.no_embedder_grad = opt.no_embedder_grad
        self.label_mask = None

        # separate loss dict
        self.loss_dict = {}

        # learning task
        self.learning_task = self.opt.task

        # id2label
        self.intent_id2label = self.opt.id2label['intent']
        self.slot_id2label = self.opt.id2label['slot']

        if opt.context_emb == 'electra':
            emb_dim = 256
        elif opt.context_emb in ['bert', 'sep_bert', 'roberta_base']:
            emb_dim = 768
        elif opt.context_emb in ['roberta_large']:
            emb_dim = 1024
        else:
            emb_dim = opt.emb_dim
        self.emb_dim = emb_dim

        # attention feature params
        if self.opt.extr_sent_reps == 'self_attn_w':
            self.sent_qw = nn.Parameter(torch.zeros(self.emb_dim, self.emb_dim))
            self.sent_kw = nn.Parameter(torch.zeros(self.emb_dim, self.emb_dim))
            self.sent_vw = nn.Parameter(torch.zeros(self.emb_dim, self.emb_dim))
            nn.init.xavier_normal_(self.sent_kw)
            nn.init.xavier_normal_(self.sent_kw)
            nn.init.xavier_normal_(self.sent_vw)
        
        if self.opt.split_metric != 'none':
            self.intent_metric = torch.nn.Linear(self.emb_dim, opt.metric_dim)
            self.slot_metric = torch.nn.Linear(self.emb_dim, opt.metric_dim)

        self.record_num = 0

        if self.opt.record_proto:
            self.record_intent_prototype = None
            self.record_slot_prototype = None

    def set_learning_task(self, task):
        if task in ['intent', 'slot_filling', 'slu']:
            self.learning_task = task
        else:
            raise TypeError('the task `{}` is not defined'.format(task))

    def get_sentence_level_reps(self, sent_test_reps, sent_support_reps, seq_test_reps, seq_support_reps, slot_test_target, slot_support_target):
        batch_size, support_size, support_len, num_tags = slot_support_target.size()
        if self.opt.extr_sent_reps == 'slot_attn':
            ''' 
            in avg_sent_reps mode, improve the importance of slot
            we should ensure that all token level reps sum should divide the number of useful tokens
            step 1: get the O label num and Slot label num
            step 2: get the average O label reps, and get the average Slot label reps
            step 3: combine the avg_O_reps and avg_Slot_reps with slot rate -- slot_attn_r
            '''
            # batch_size, support_size, support_len = support_token_ids.size()
            expand_slot_test_target = slot_test_target.unsqueeze(1).repeat(1, support_size,
                                                                           1)  # (batch_size, support_size, test_len)
            sum_slot_support_target = self.onehot2label_id(slot_support_target).to(slot_support_target.device)

            ''' average (average slot reps + average O reps) '''
            # for test
            test_slot_output_mask = (expand_slot_test_target > 1).float()  # (batch_size, support_size, test_len)
            test_O_output_mask = (expand_slot_test_target == 1).float()  # (batch_size, support_size, test_len)
            slot_seq_test_reps = torch.sum(seq_test_reps * test_slot_output_mask.unsqueeze(-1), dim=-2)
            slot_nums = torch.sum(test_slot_output_mask, dim=-1).unsqueeze(-1)

            # if slot num == 0, then `nan` will appear
            # but, if slot num == 0, the slot_seq_test_reps == 0
            # so make slot num == 1
            no_zero_add = (slot_nums == 0).float()
            slot_nums = slot_nums + no_zero_add
            slot_sent_test_reps = torch.div(slot_seq_test_reps, slot_nums)

            O_seq_test_reps = torch.sum(seq_test_reps * test_O_output_mask.unsqueeze(-1), dim=-2)
            O_nums = torch.sum(test_O_output_mask, dim=-1).unsqueeze(-1)

            # if O num == 0, then `nan` will appear
            # but, if O num == 0, the O_seq_test_reps == 0
            # so make O num == 1
            no_zero_add = (O_nums == 0).float()
            O_nums = O_nums + no_zero_add
            O_sent_test_reps = torch.div(O_seq_test_reps, O_nums)

            # sent_test_reps = (slot_sent_test_reps + O_sent_test_reps) / 2
            sent_test_reps = self.opt.slot_attn_r * slot_sent_test_reps + (1 - self.opt.slot_attn_r) * O_sent_test_reps
            sent_test_reps = sent_test_reps.unsqueeze(-2)  # (batch_size, support_size, 1, emb_dim)

            # for support
            support_slot_output_mask = (sum_slot_support_target > 1).float()
            support_O_output_mask = (sum_slot_support_target == 1).float()
            slot_seq_support_reps = torch.sum(seq_support_reps * support_slot_output_mask.unsqueeze(-1), dim=-2)
            slot_support_nums = torch.sum(support_slot_output_mask, dim=-1).unsqueeze(-1)

            # if slot num == 0, then `nan` will appear
            # but, if slot num == 0, the slot_seq_support_reps == 0
            # so make slot num == 1
            no_zero_add = (slot_support_nums == 0).float()
            slot_support_nums = slot_support_nums + no_zero_add
            slot_sent_support_reps = torch.div(slot_seq_support_reps, slot_support_nums)
            O_seq_support_reps = torch.sum(seq_support_reps * support_O_output_mask.unsqueeze(-1), dim=-2)
            O_support_nums = torch.sum(support_O_output_mask, dim=-1).unsqueeze(-1)

            # if O num == 0, then `nan` will appear
            # but, if O num == 0, the O_seq_support_reps == 0
            # so make O num == 1
            no_zero_add = (O_support_nums == 0).float()
            O_support_nums = O_support_nums + no_zero_add
            O_sent_support_reps = torch.div(O_seq_support_reps, O_support_nums)

            # sent_support_reps = (slot_sent_support_reps + O_sent_support_reps) / 2
            sent_support_reps = self.opt.slot_attn_r * slot_sent_support_reps + (1 - self.opt.slot_attn_r) * O_sent_support_reps
            sent_support_reps = sent_support_reps.unsqueeze(-2)  # (batch_size, support_size, 1, emb_dim)

        elif self.opt.extr_sent_reps in ['self_attn', 'self_attn_w']:
            # test
            batch_size, support_size, test_len, emb_dim = seq_test_reps.size()
            re_seq_test_reps = seq_test_reps.contiguous().view(batch_size * support_size, test_len, emb_dim)
            if self.opt.extr_sent_reps == 'self_attn_w':
                q_test_reps = torch.bmm(re_seq_test_reps, self.sent_qw.unsqueeze(0).repeat(batch_size * support_size, 1, 1))
                k_test_reps = torch.bmm(re_seq_test_reps, self.sent_kw.unsqueeze(0).repeat(batch_size * support_size, 1, 1))
                v_test_reps = torch.bmm(re_seq_test_reps, self.sent_vw.unsqueeze(0).repeat(batch_size * support_size, 1, 1))
            else:
                q_test_reps, k_test_reps, v_test_reps = re_seq_test_reps, re_seq_test_reps, re_seq_test_reps
            # (batch_size * support_size, test_len, test_len)
            test_score_matrix = torch.bmm(q_test_reps, k_test_reps.transpose(2, 1))
            test_score_matrix = torch.div(test_score_matrix, self.emb_dim ** 0.5)
            attn_seq_test_score = torch.softmax(test_score_matrix, dim=-1)
            sent_test_reps = torch.mean(
                torch.sum(
                    v_test_reps.unsqueeze(-2).repeat(1, 1, test_len, 1) * attn_seq_test_score.unsqueeze(-1),
                    dim=-2),  # (batch_size * support_size, test_len, emb_dim)
                dim=-2)  # (batch_size * support_size, emb_dim)
            sent_test_reps = sent_test_reps.unsqueeze(-2)  # (batch_size * support_size, 1, emb_dim)
            sent_test_reps = sent_test_reps.contiguous().view(batch_size, support_size, 1, -1)

            # support
            batch_size, support_size, support_len, emb_dim = seq_support_reps.size()
            re_seq_support_reps = seq_support_reps.contiguous().view(batch_size * support_size, support_len, emb_dim)
            if self.opt.extr_sent_reps == 'self_attn_w':
                q_spt_reps = torch.bmm(re_seq_support_reps, self.sent_qw.unsqueeze(0).repeat(batch_size * support_size, 1, 1))
                k_spt_reps = torch.bmm(re_seq_support_reps, self.sent_kw.unsqueeze(0).repeat(batch_size * support_size, 1, 1))
                v_spt_reps = torch.bmm(re_seq_support_reps, self.sent_vw.unsqueeze(0).repeat(batch_size * support_size, 1, 1))
            else:
                q_spt_reps, k_spt_reps, v_spt_reps = re_seq_support_reps, re_seq_support_reps, re_seq_support_reps
            # (batch_size, support_size, support_len, support_len)
            support_score_matrix = torch.bmm(q_spt_reps, k_spt_reps.transpose(2, 1))
            support_score_matrix = torch.div(support_score_matrix, self.emb_dim ** 0.5)
            attn_seq_support_score = torch.softmax(support_score_matrix, dim=-1)
            sent_support_reps = torch.mean(
                torch.sum(
                    v_spt_reps.unsqueeze(-2).repeat(1, 1, support_len, 1) * attn_seq_support_score.unsqueeze(-1),
                    dim=-2),  # (batch_size * support_size, support_len, emb_dim)
                dim=-2)  # (batch_size * support_size, emb_dim)
            sent_support_reps = sent_support_reps.unsqueeze(-2)  # (batch_size * support_size, 1, emb_dim)
            sent_support_reps = sent_support_reps.contiguous().view(batch_size, support_size, 1, -1)

        elif self.opt.extr_sent_reps == 'coarse_slot_self_attn':
            ''' use slot token as query, label the importance of each token '''
            # batch_size, support_size, support_len = support_token_ids.size()
            expand_slot_test_target = slot_test_target.unsqueeze(1).repeat(1, support_size,
                                                                           1)  # (batch_size, support_size, test_len)
            sum_slot_support_target = self.onehot2label_id(slot_support_target).to(slot_support_target.device)

            ''' average (average slot reps + average O reps) '''
            # for test
            batch_size, support_size, test_len, emb_dim = seq_test_reps.size()
            test_slot_output_mask = (expand_slot_test_target > 1).float()  # (batch_size, support_size, test_len)
            # (batch_size, support_size, dim)
            avg_slot_test_reps = torch.mean(seq_test_reps * test_slot_output_mask.unsqueeze(-1), dim=-2)  # as query
            # (batch_size, support_size, test_len)
            test_score_matrix = torch.bmm(avg_slot_test_reps.contiguous().view(batch_size * support_size, 1, -1),
                                          seq_test_reps.contiguous().view(batch_size * support_size, test_len, -1
                                                                          ).transpose(2, 1))
            test_score_matrix = test_score_matrix.contiguous().view(batch_size, support_size, test_len)
            test_score_matrix = torch.div(test_score_matrix, self.emb_dim ** 0.5)
            attn_seq_test_score = torch.softmax(test_score_matrix, dim=-1)
            sent_test_reps = torch.mean(attn_seq_test_score.unsqueeze(-1) * seq_test_reps, dim=-2)
            sent_test_reps = sent_test_reps.contiguous().view(batch_size, support_size, 1, -1)

            # for support
            batch_size, support_size, support_len, emb_dim = seq_support_reps.size()
            spt_slot_output_mask = (sum_slot_support_target > 1).float()
            avg_slot_spt_reps = torch.mean(seq_support_reps * spt_slot_output_mask.unsqueeze(-1), dim=-2)  # as query

            spt_score_matrix = torch.bmm(avg_slot_spt_reps.contiguous().view(batch_size * support_size, 1, -1),
                                          seq_support_reps.contiguous().view(batch_size * support_size, support_len, -1
                                                                             ).transpose(2, 1))
            spt_score_matrix = spt_score_matrix.contiguous().view(batch_size, support_size, support_len)
            spt_score_matrix = torch.div(spt_score_matrix, self.emb_dim ** 0.5)
            attn_seq_spt_score = torch.softmax(spt_score_matrix, dim=-1)
            sent_support_reps = torch.mean(attn_seq_spt_score.unsqueeze(-1) * seq_support_reps, dim=-2)
            sent_support_reps = sent_support_reps.contiguous().view(batch_size, support_size, 1, -1)

        elif self.opt.extr_sent_reps == 'fine_slot_self_attn':
            ''' use slot token as query, label the importance of each token '''
            # batch_size, support_size, support_len = support_token_ids.size()
            expand_slot_test_target = slot_test_target.unsqueeze(1).repeat(1, support_size,
                                                                           1)  # (batch_size, support_size, test_len)
            sum_slot_support_target = self.onehot2label_id(slot_support_target).to(slot_support_target.device)

            ''' average (average slot reps + average O reps) '''
            # for test
            batch_size, support_size, test_len, emb_dim = seq_test_reps.size()
            test_slot_output_mask = (expand_slot_test_target > 1)  # (batch_size, support_size, test_len)
            all_sent_test_reps = []
            # (batch_size, support_size, dim)
            for b_idx in range(batch_size):
                for s_idx in range(support_size):
                    one_slot_test_reps = torch.masked_select(seq_test_reps[b_idx, s_idx],
                                                             test_slot_output_mask[b_idx, s_idx].unsqueeze(-1)).to(seq_test_reps)
                    one_slot_test_reps = one_slot_test_reps.contiguous().view(-1, emb_dim)  # as query
                    slot_num = one_slot_test_reps.size(0)
                    if slot_num:
                        # (slot_num, test_len)
                        one_test_score_matrix = torch.mm(one_slot_test_reps, seq_test_reps[b_idx, s_idx].transpose(1, 0))
                        one_test_score_matrix = torch.div(one_test_score_matrix, self.emb_dim ** 0.5)
                        one_attn_seq_test_score = torch.softmax(one_test_score_matrix, dim=-1)
                        one_sent_test_reps = torch.mean(one_attn_seq_test_score.unsqueeze(-1) *
                                                        seq_test_reps[b_idx, s_idx].unsqueeze(0).repeat(slot_num, 1, 1), dim=-2)
                        one_sent_test_reps = torch.mean(one_sent_test_reps, dim=-2)
                    else:
                        one_sent_test_reps = torch.zeros(emb_dim).to(seq_support_reps)
                    all_sent_test_reps.append(one_sent_test_reps)
            sent_test_reps = torch.stack(all_sent_test_reps, dim=0).contiguous().view(batch_size, support_size, -1, emb_dim)

            # for support
            batch_size, support_size, support_len, emb_dim = seq_support_reps.size()
            spt_slot_output_mask = (sum_slot_support_target > 1)
            all_sent_spt_reps = []
            for b_idx in range(batch_size):
                for s_idx in range(support_size):

                    one_slot_spt_reps = torch.masked_select(seq_support_reps[b_idx, s_idx],
                                                            spt_slot_output_mask[b_idx, s_idx].unsqueeze(-1)).to(seq_support_reps)
                    one_slot_spt_reps = one_slot_spt_reps.contiguous().view(-1, emb_dim)  # as query
                    slot_num = one_slot_spt_reps.size(0)
                    if slot_num:
                        # (slot_num, test_len)
                        one_spt_score_matrix = torch.mm(one_slot_spt_reps, seq_support_reps[b_idx, s_idx].transpose(1, 0))
                        one_spt_score_matrix = torch.div(one_spt_score_matrix, self.emb_dim ** 0.5)
                        one_attn_seq_spt_score = torch.softmax(one_spt_score_matrix, dim=-1)
                        one_sent_support_reps = torch.mean(one_attn_seq_spt_score.unsqueeze(-1) *
                                                           seq_support_reps[b_idx, s_idx].unsqueeze(0).repeat(slot_num, 1, 1), dim=-2)
                        one_sent_support_reps = torch.mean(one_sent_support_reps, dim=-2)
                    else:
                        one_sent_support_reps = torch.zeros(emb_dim).to(seq_support_reps)
                    all_sent_spt_reps.append(one_sent_support_reps)
            sent_support_reps = torch.stack(all_sent_spt_reps, dim=0).contiguous().view(batch_size, support_size, -1, emb_dim)
        elif self.opt.extr_sent_reps == 'none':
            pass
        else:
            raise NotImplementedError
        return sent_test_reps, sent_support_reps

    def get_reps_after_split_metric(self, sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps):
        atv_lst = self.opt.metric_activation.split('-')
        intent_activation = atv_lst[0]
        slot_activation = atv_lst[0] if len(atv_lst) == 1 else atv_lst[1]

        if self.opt.split_metric in ['intent', 'both']:
            # intent metric space
            sent_test_reps = self.intent_metric(sent_test_reps)
            sent_support_reps = self.intent_metric(sent_support_reps)
            # activation
            if intent_activation == 'relu':
                sent_test_reps = F.relu(sent_test_reps)
                sent_support_reps = F.relu(sent_support_reps)
            elif intent_activation == 'sigmoid':
                sent_test_reps = torch.sigmoid(sent_test_reps)
                sent_support_reps = torch.sigmoid(sent_support_reps)
            elif intent_activation == 'tanh':
                sent_test_reps = torch.tanh(sent_test_reps)
                sent_support_reps = torch.tanh(sent_support_reps)
            elif intent_activation == 'none':
                pass
            else:
                raise TypeError('the metric_activation {} is not defined'.format(self.opt.metric_activation))

        if self.opt.split_metric in ['slot', 'both']:
            # slot metric space
            seq_test_reps = self.slot_metric(seq_test_reps)
            seq_support_reps = self.slot_metric(seq_support_reps)
            # activation
            if slot_activation == 'relu':
                seq_test_reps = F.relu(seq_test_reps)
                seq_support_reps = F.relu(seq_support_reps)
            elif slot_activation == 'sigmoid':
                seq_test_reps = torch.sigmoid(seq_test_reps)
                seq_support_reps = torch.sigmoid(seq_support_reps)
            elif slot_activation == 'tanh':
                seq_test_reps = torch.tanh(seq_test_reps)
                seq_support_reps = torch.tanh(seq_support_reps)
            elif slot_activation == 'none':
                pass
            else:
                raise TypeError('the metric_activation {} is not defined'.format(self.opt.metric_activation))
        return sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps

    def add_contrastive_loss(self, intent_support_target, slot_support_target, intent_prototype, slot_prototype):
        regular_loss = 0.0
        if self.learning_task == 'slu':
            regular_loss = 0.
            if self.opt.slu_regular == 'contrastive':
                regular_loss = self.cal_contrastive_loss(slot_support_target, intent_support_target,
                                                         self.slot_id2label, self.intent_id2label,
                                                         intent_prototype, slot_prototype)
                if self.opt.with_homo:
                    regular_loss += self.cal_same_proto_loss(intent_prototype, slot_prototype)
            elif self.opt.slu_regular == 'triplet':
                regular_loss = self.cal_triplet_loss(slot_support_target, intent_support_target,
                                                     self.slot_id2label, self.intent_id2label,
                                                     intent_prototype, slot_prototype)
                if self.opt.with_homo:
                    regular_loss += self.cal_same_proto_loss(intent_prototype, slot_prototype)
                elif self.opt.with_homo2:
                    regular_loss += self.cal_same_proto_loss2(intent_prototype, slot_prototype)
            elif self.opt.slu_regular == 'triplet2':
                regular_loss = self.cal_triplet2_loss(slot_support_target, intent_support_target,
                                                      self.slot_id2label, self.intent_id2label,
                                                      intent_prototype, slot_prototype)
                if self.opt.with_homo:
                    regular_loss += self.cal_same_proto_loss(intent_prototype, slot_prototype)
                elif self.opt.with_homo2:
                    regular_loss += self.cal_same_proto_loss2(intent_prototype, slot_prototype)
            elif self.opt.slu_regular == 'strict_triplet2':
                regular_loss = self.cal_strict_triplet2_loss(slot_support_target, intent_support_target,
                                                             self.slot_id2label, self.intent_id2label,
                                                             intent_prototype, slot_prototype)
                if self.opt.with_homo:
                    regular_loss += self.cal_same_proto_loss(intent_prototype, slot_prototype)
                elif self.opt.with_homo2:
                    regular_loss += self.cal_same_proto_loss2(intent_prototype, slot_prototype)
            elif self.opt.slu_regular == 'triplet_semi_hard':
                regular_loss = self.cal_triplet_semi_hard_loss(slot_support_target, intent_support_target,
                                                               self.slot_id2label, self.intent_id2label,
                                                               intent_prototype, slot_prototype)
                if self.opt.with_homo:
                    regular_loss += self.cal_same_proto_loss(intent_prototype, slot_prototype)
                elif self.opt.with_homo2:
                    regular_loss += self.cal_same_proto_loss2(intent_prototype, slot_prototype)
            elif self.opt.slu_regular == 'strict_triplet_semi_hard':
                regular_loss = self.cal_strict_triplet_semi_hard_loss(slot_support_target, intent_support_target,
                                                                      self.slot_id2label, self.intent_id2label,
                                                                      intent_prototype, slot_prototype)
                if self.opt.with_homo:
                    regular_loss += self.cal_same_proto_loss(intent_prototype, slot_prototype)
                elif self.opt.with_homo2:
                    regular_loss += self.cal_same_proto_loss2(intent_prototype, slot_prototype)
            elif self.opt.slu_regular == 'homo_contrastive':
                regular_loss = self.cal_same_proto_loss(intent_prototype, slot_prototype)
            elif self.opt.slu_regular == 'homo_triplet2':
                regular_loss = self.cal_same_proto_loss2(intent_prototype, slot_prototype)
            elif self.opt.slu_regular == 'none':
                pass
            else:
                raise TypeError('the intent_slot_regular: {} is not defined'.format(self.opt.slu_regular))

        return regular_loss

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: (batch_size, test_len)
        :param intent_test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: (batch_size, support_size, support_len)
        :param intent_support_output_mask: (batch_size, support_size, support_len)
        :param slot_test_target: index targets (batch_size, test_len)
        :param slot_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: index targets (batch_size, test_len)
        :param intent_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :return:
        """
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        ''' get sentence level representation '''
        sent_test_reps, sent_support_reps = self.get_sentence_level_reps(sent_test_reps, sent_support_reps,
                                                                         seq_test_reps, seq_support_reps,
                                                                         slot_test_target, slot_support_target)

        ''' metric space controller '''
        sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps = self.get_reps_after_split_metric(
            sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps)

        ''' getting prototype'''
        intent_prototype, slot_prototype = None, None
        if self.opt.task in ['intent', 'slu']:
            intent_prototype = self.intent_decoder.cal_prototype(sent_test_reps, intent_test_output_mask,
                                                                 sent_support_reps, intent_support_output_mask,
                                                                 intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_prototype = self.slot_decoder.cal_prototype(seq_test_reps, slot_test_output_mask,
                                                             seq_support_reps, slot_support_output_mask,
                                                             slot_test_target, slot_support_target)
        ''' getting emission '''
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(sent_test_reps, intent_test_output_mask,
                                                               sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target)
            intent_prototype = self.intent_decoder.get_prototype()  # (batch_size, num_tags, emd_dim)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(seq_test_reps, slot_test_output_mask,
                                                           seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target)
            slot_prototype = self.slot_decoder.get_prototype()  # (batch_size, num_tags, emd_dim)

        if self.training:
            loss = 0.
            if self.learning_task in ['intent', 'slu']:
                intent_loss = self.intent_decoder(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                  intent_support_output_mask, intent_test_target,
                                                  intent_support_target, self.label_mask)
                loss += intent_loss
                self.loss_dict['intent'] = intent_loss
            if self.learning_task in ['slot_filling', 'slu']:
                slot_loss = self.slot_decoder(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                              slot_support_output_mask, slot_test_target, slot_support_target,
                                              self.label_mask)
                loss += self.opt.loss_rate * slot_loss
                self.loss_dict['slot'] = slot_loss

            regular_loss = self.add_contrastive_loss(intent_support_target, slot_support_target, intent_prototype,
                                                     slot_prototype)
            loss += self.opt.slu_regular_rate * regular_loss
            return loss
        else:
            # '''store visualization embedding'''
            if self.opt.record_proto:
                self.record_intent_prototype = intent_prototype.detach().cpu()
                self.record_slot_prototype = slot_prototype.detach().cpu()
            intent_preds, slot_preds = None, None
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                          intent_support_output_mask, intent_test_target,
                                                          intent_support_target, self.label_mask)
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                                      slot_support_output_mask, slot_test_target,
                                                      slot_support_target, self.label_mask)
            return {'slot': slot_preds, 'intent': intent_preds}

    def get_loss_dict(self):
        return self.loss_dict

    ''' 4 methods for fix parameters '''
    def childs(self, m):
        return m if isinstance(m, (list, tuple)) else list(m.children())

    def set_trainable_attr(self, m, b):
        m.trainable = b
        for p in m.parameters():
            p.requires_grad = b

    def apply_leaf(self, m, f):
        c = self.childs(m)
        if isinstance(m, nn.Module):
            f(m)
        if len(c) > 0:
            for l in c:
                self.apply_leaf(l, f)

    def set_trainable(self, l, b):
        self.apply_leaf(l, lambda m: self.set_trainable_attr(m, b))

    def get_context_reps(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            use_cls: bool = False,
    ):
        if self.no_embedder_grad:
            self.context_embedder.eval()  # to avoid the dropout effect of reps model
            self.context_embedder.requires_grad = False
        else:
            self.context_embedder.train()  # to avoid the dropout effect of reps model
            self.context_embedder.requires_grad = True
            if self.opt.fix_encoder_layer != 0:
                self.set_trainable(self.context_embedder.embedder.embeddings, False)
                for i in range(self.opt.fix_encoder_layer):
                    self.set_trainable(self.context_embedder.embedder.encoder.layer[i], False)

        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.context_embedder(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids,
            support_segment_ids,
            support_nwp_index, support_input_mask, use_cls=use_cls
        )
        if self.no_embedder_grad:
            seq_test_reps = seq_test_reps.detach()  # detach the reps part from graph
            seq_support_reps = seq_support_reps.detach()  # detach the reps part from graph
            sent_test_reps = sent_test_reps.detach()  # detach the reps part from graph
            sent_support_reps = sent_support_reps.detach()  # detach the reps part from graph
        return seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps

    def onehot2label_id(self, inputs):
        origin_size = list(inputs.size())
        num_tags = origin_size[-1]
        inputs = inputs.view(-1, num_tags)
        after_size = inputs.size()[0]
        outputs = []
        for a_idx in range(after_size):
            tags = inputs[a_idx].tolist()
            if sum(tags) > 0:
                label_id = tags.index(1)
            else:  # the padding
                label_id = 0
            outputs.append(label_id)
        origin_size[-1] = 1
        outputs = torch.tensor(outputs, dtype=torch.long)
        outputs = outputs.reshape(*origin_size).squeeze(-1)
        return outputs

    def get_support_intent2slot_mask(self, intent_support_target, slot_support_target, intent_id2label, slot_id2label):
        """
        get the intent2slot relation mask from support set
        the [PAD] label has no relation between intent and slot, so we should set relative column and row to 0
        """
        batch_intent2slot_map = self.cal_intent2slot_map_from_support(
            intent_support_target, slot_support_target, intent_id2label, slot_id2label)
        intent2slot_mask = (batch_intent2slot_map != 0).float()
        return intent2slot_mask

    def cal_intent2slot_map_from_support(self, intent_support_target, slot_support_target, intent_id2label, slot_id2label):
        intent_support_target = self.onehot2label_id(intent_support_target)
        slot_support_target = self.onehot2label_id(slot_support_target)
        # get the size info
        batch_size, support_size, test_len = slot_support_target.size()

        batch_intent2slot_map = []
        for b_idx in range(batch_size):
            intent2slot_map = torch.zeros(len(intent_id2label), len(slot_id2label))
            for s_idx in range(support_size):
                intent_label_id = intent_support_target[b_idx][s_idx][0]
                for slot_label_id in slot_support_target[b_idx][s_idx]:
                    if slot_id2label[slot_label_id.item()] not in ['O', '[PAD]']:
                        intent2slot_map[intent_label_id][slot_label_id] += 1
            batch_intent2slot_map.append(intent2slot_map)
        batch_intent2slot_map = torch.stack(batch_intent2slot_map, dim=0)
        return batch_intent2slot_map

    def get_slot2intent_dict(self, slot_support_target, intent_support_target, slot_id2label, intent_id2label):
        slot_support_target = self.onehot2label_id(slot_support_target)
        intent_support_target = self.onehot2label_id(intent_support_target)
        # get the size info
        batch_size, support_size, test_len = slot_support_target.size()

        intent_support_target = intent_support_target.view(batch_size, -1).tolist()

        batch_slot2intent_lst = []
        for b_idx in range(batch_size):
            slot_id2intent_lst = {slot_id: Counter() for slot_id in slot_id2label}
            slot2intent_lst = {slot_id2label[slot_id]: Counter() for slot_id in slot_id2label}
            for s_idx in range(support_size):
                intent_label_id = intent_support_target[b_idx][s_idx]
                for slot_label_id in slot_support_target[b_idx][s_idx]:
                    if slot_id2label[slot_label_id.item()] not in ['O', '[PAD]']:
                        slot_id2intent_lst[slot_label_id.item()][intent_id2label[intent_label_id]] += 1
                        slot2intent_lst[slot_id2label[slot_label_id.item()]][intent_id2label[intent_label_id]] += 1
            batch_slot2intent_lst.append(slot_id2intent_lst)
        return batch_slot2intent_lst

    def expand_intent_emission_to_slot(self, slot_support_target, intent_support_target, slot_id2label, intent_id2label,
                                       intent_emission) -> torch.Tensor:
        """
        get intent prototype reps by deriving from slot prototype reps, via support set statistic
        :param slot_support_target: (batch_size, support_size, test_len)
        :param intent_support_target: (batch_size, support_size, 1)
        :param slot_id2label:
        :param intent_id2label:
        :param intent_emission: (batch_size, test_len, no_pad_intent_num_tags)
        :return: (batch_size, test_len, no_pad_intent_num_tags)
        """
        slot_support_target = self.onehot2label_id(slot_support_target)
        intent_support_target = self.onehot2label_id(intent_support_target)
        # get the size info
        batch_size, support_size, test_len = slot_support_target.size()
        # the test_len == 1, so remove it
        intent_emission = intent_emission.view(batch_size, -1)

        intent_support_target = intent_support_target.view(batch_size, -1).tolist()

        batch_expand_intent_emission = []
        for b_idx in range(batch_size):
            slot2intent_lst = {slot_id: Counter() for slot_id in slot_id2label}
            for s_idx in range(support_size):
                intent_label_id = intent_support_target[b_idx][s_idx]
                for slot_label_id in slot_support_target[b_idx][s_idx]:
                    if slot_id2label[slot_label_id.item()] not in ['O', '[PAD]']:
                        slot2intent_lst[slot_label_id.item()][intent_label_id] += 1

            expand_intent_emission = []
            for slot_id in slot_id2label:
                if slot_id == 0:  # the [PAD] label is removed
                    continue
                if len(slot2intent_lst[slot_id]) == 0:
                    expand_intent_emission.append(torch.tensor(0, dtype=torch.float).to(intent_emission.device))
                else:
                    all_intent_label_ids = list(slot2intent_lst[slot_id].keys())
                    all_intent_label_ids = [label_id - 1 for label_id in all_intent_label_ids]  # remove [pad] label
                    mean_val = torch.mean(intent_emission[b_idx][all_intent_label_ids], dim=0)
                    expand_intent_emission.append(mean_val)
            expand_intent_emission = torch.stack(expand_intent_emission, dim=0)
            batch_expand_intent_emission.append(expand_intent_emission)

        batch_expand_intent_emission = torch.stack(batch_expand_intent_emission, dim=0)
        batch_expand_intent_emission = batch_expand_intent_emission.unsqueeze(1)  # add test_len == 1
        return batch_expand_intent_emission

    def expand_slot_emission_to_intent(self, slot_support_target, intent_support_target, slot_id2label, intent_id2label,
                                       slot_emission) -> torch.Tensor:
        """
        get intent prototype reps by deriving from slot prototype reps, via support set statistic
        :param slot_support_target: (batch_size, support_size, test_len)
        :param intent_support_target: (batch_size, support_size, 1)
        :param slot_id2label:
        :param intent_id2label:
        :param slot_emission: (batch_size, test_len, no_pad_intent_num_tags)
        :return: (batch_size, test_len, no_pad_intent_num_tags)
        """
        slot_support_target = self.onehot2label_id(slot_support_target)
        intent_support_target = self.onehot2label_id(intent_support_target)

        batch_size, support_size, test_len = slot_support_target.size()

        intent_support_target = intent_support_target.view(batch_size, -1).tolist()

        batch_expand_slot_emission = []
        for b_idx in range(batch_size):
            intent2slot_lst = {intent_id: Counter() for intent_id in intent_id2label}
            for s_idx in range(support_size):
                intent_label_id = intent_support_target[b_idx][s_idx]
                for slot_label_id in slot_support_target[b_idx][s_idx]:
                    if slot_id2label[slot_label_id.item()] not in ['O', '[PAD]']:
                        intent2slot_lst[intent_label_id][slot_label_id.item()] += 1

            expand_slot_emission = []
            for intent_id in intent_id2label:
                if intent_id == 0:  # the [PAD] label is removed
                    continue
                if len(intent2slot_lst[intent_id]) == 0:
                    expand_slot_emission.append(torch.tensor(0, dtype=torch.float).to(slot_emission.device))
                else:
                    all_slot_label_ids = list(intent2slot_lst[intent_id].keys())
                    all_slot_label_ids = [label_id - 1 for label_id in all_slot_label_ids]  # remove [pad] label
                    mean_val = torch.mean(slot_emission[b_idx][:, all_slot_label_ids])
                    expand_slot_emission.append(mean_val)
            expand_slot_emission = torch.stack(expand_slot_emission, dim=0)
            batch_expand_slot_emission.append(expand_slot_emission)

        batch_expand_slot_emission = torch.stack(batch_expand_slot_emission, dim=0)
        batch_expand_slot_emission = batch_expand_slot_emission.unsqueeze(1)
        return batch_expand_slot_emission

    def cal_contrastive_loss(self, slot_support_target, intent_support_target, slot_id2label, intent_id2label,
                             intent_prototype, slot_prototype):
        """the related intent-slot proto should be close, the unrelated intent-slot proto should be far way"""
        intent2slot_mask = self.get_support_intent2slot_mask(
            intent_support_target, slot_support_target, intent_id2label, slot_id2label)
        intent2slot_mask = intent2slot_mask.to(intent_prototype.device)

        # (batch_size, intent_tag_num, slot_tag_num, dim)
        distance = intent_prototype.unsqueeze(-2) - slot_prototype.unsqueeze(1)
        dw = torch.norm(distance, p=2, dim=-1)  # (batch_size, intent_tag_num, slot_tag_num)

        zero_v = torch.zeros_like(dw).to(dw.device)
        distance = intent2slot_mask * torch.pow(dw, 2) / 2 + (1 - intent2slot_mask) * torch.pow(torch.max(zero_v, self.opt.margin - dw), 2) / 2

        distance = torch.mean(distance)

        if torch.isnan(distance).sum():
            raise ValueError('there is nan in contrastive')

        return distance

    def cal_triplet_loss(self, slot_support_target, intent_support_target, slot_id2label, intent_id2label,
                         intent_prototype, slot_prototype):
        """use triplet loss to push away the intent proto and pull the slot proto close to the related intent proto"""
        intent2slot_mask = self.get_support_intent2slot_mask(
            intent_support_target, slot_support_target, intent_id2label, slot_id2label)
        intent2slot_mask = intent2slot_mask.to(intent_prototype.device)

        distance = intent_prototype.unsqueeze(-2) - slot_prototype.unsqueeze(1)
        dw = torch.norm(distance, p=2, dim=-1)  # (batch_size, intent_tag_num, slot_tag_num)

        zero_v = torch.zeros_like(dw).to(dw.device)
        distance = torch.max(zero_v, intent2slot_mask * dw - (1 - intent2slot_mask) * dw + self.opt.margin)
        distance = torch.mean(distance)

        return distance

    def cal_triplet2_loss(self, slot_support_target, intent_support_target, slot_id2label, intent_id2label,
                          intent_prototype, slot_prototype):
        """ use square operation in calculating distance """
        intent2slot_mask = self.get_support_intent2slot_mask(
            intent_support_target, slot_support_target, intent_id2label, slot_id2label)
        intent2slot_mask = intent2slot_mask.to(intent_prototype.device)

        distance = intent_prototype.unsqueeze(-2) - slot_prototype.unsqueeze(1)
        dw = torch.norm(distance, p=2, dim=-1)  # (batch_size, intent_tag_num, slot_tag_num)

        zero_v = torch.zeros_like(dw).to(dw.device)
        distance = torch.max(zero_v, intent2slot_mask * torch.pow(dw, 2) - (1 - intent2slot_mask) * torch.pow(dw, 2)
                             + self.opt.margin)
        distance = torch.mean(distance)

        return distance

    def cal_strict_triplet2_loss(self, slot_support_target, intent_support_target, slot_id2label, intent_id2label,
                                 intent_prototype, slot_prototype):
        """use strict triplet loss with square operation"""
        intent2slot_mask = self.get_support_intent2slot_mask(
            intent_support_target, slot_support_target, intent_id2label, slot_id2label)
        intent2slot_mask = intent2slot_mask.to(intent_prototype.device)

        distance = intent_prototype.unsqueeze(-2) - slot_prototype.unsqueeze(1)
        dw = torch.norm(distance, p=2, dim=-1)  # (batch_size, intent_tag_num, slot_tag_num)

        postive_dw = intent2slot_mask * torch.pow(dw, 2)
        negative_dw = (1 - intent2slot_mask) * torch.pow(dw, 2)
        # (batch_size, intent_tag_num, slot_tag_num, slot_tag_num)
        useful_mask = intent2slot_mask.unsqueeze(-1) * (1 - intent2slot_mask).unsqueeze(-2)
        distance = useful_mask * (postive_dw.unsqueeze(-1) - negative_dw.unsqueeze(-2))

        zero_v = torch.zeros_like(distance).to(distance.device)
        distance = torch.max(zero_v, distance + self.opt.margin)
        distance = torch.mean(distance)

        return distance

    def cal_triplet_semi_hard_loss(self, slot_support_target, intent_support_target, slot_id2label, intent_id2label,
                                   intent_prototype, slot_prototype):
        """
        use square operation in calculating distance,
        and use semi-hard negative sampling strategy, k*(i, j) = arg min D_i,k ^2 s.t. D_i,k ^2 > D_i,j ^2
        """
        batch_size = slot_support_target.size(0)
        intent2slot_mask = self.get_support_intent2slot_mask(
            intent_support_target, slot_support_target, intent_id2label, slot_id2label)
        # (batch_size, intent_tag_num*slot_tag_num)
        intent2slot_mask = intent2slot_mask.view(batch_size, -1).to(intent_prototype.device)

        distance = intent_prototype.unsqueeze(-2) - slot_prototype.unsqueeze(1)
        dw = torch.norm(distance, p=2, dim=-1)  # (batch_size, intent_tag_num, slot_tag_num)
        dw = dw.view(batch_size, -1)  # (batch_size, intent_tag_num*slot_tag_num)

        # semi-hard negative sampling strategy, find the smallest negative sample which is bigger than positive sample
        negative_mask = 1 - intent2slot_mask
        positive_dw = intent2slot_mask * torch.pow(dw, 2)  # positive sample
        # negative sample, set the non-negative sample a big value
        negative_dw = negative_mask * torch.pow(dw, 2) + intent2slot_mask * 1e9
        # (batch_size, intent_tag_num*slot_tag_num, intent_tag_num*slot_tag_num)
        semi_hard_negative_dw = negative_dw.unsqueeze(-2) - positive_dw.unsqueeze(-1)
        # should remove the hard samples, whose distance is smaller than positive samples
        # we can set hard samples as a big value
        hard_mask = (semi_hard_negative_dw <= 0).float()
        semi_hard_negative_dw += hard_mask * 1e9
        # get semi-hard negative samples, whose number equal to positive samples
        semi_hard_negative_dw = torch.min(semi_hard_negative_dw, dim=-1)[0]
        # just need the positive sample in dimension 1
        semi_hard_negative_dw = intent2slot_mask * semi_hard_negative_dw

        zero_v = torch.zeros_like(dw).to(dw.device)
        distance = torch.max(zero_v, positive_dw - semi_hard_negative_dw + self.opt.margin)
        distance = torch.mean(distance)

        return distance

    def cal_strict_triplet_semi_hard_loss(self, slot_support_target, intent_support_target, slot_id2label,
                                          intent_id2label, intent_prototype, slot_prototype):
        """
        use strict semi-hard sampling strategy in triplet loss
        """
        intent2slot_mask = self.get_support_intent2slot_mask(
            intent_support_target, slot_support_target, intent_id2label, slot_id2label)
        # (batch_size, intent_tag_num, slot_tag_num)
        intent2slot_mask = intent2slot_mask.to(intent_prototype.device)

        distance = intent_prototype.unsqueeze(-2) - slot_prototype.unsqueeze(1)
        dw = torch.norm(distance, p=2, dim=-1)  # (batch_size, intent_tag_num, slot_tag_num)

        # semi-hard negative sampling strategy, find the smallest negative sample which is bigger than positive sample
        postive_dw = intent2slot_mask * torch.pow(dw, 2)
        negative_dw = (1 - intent2slot_mask) * torch.pow(dw, 2)
        # (batch_size, intent_tag_num, slot_tag_num, slot_tag_num)
        useful_mask = intent2slot_mask.unsqueeze(-1) * (1 - intent2slot_mask).unsqueeze(-2)
        distance = useful_mask * (postive_dw.unsqueeze(-1) - negative_dw.unsqueeze(-2))
        # set unuseful sample and hard negative sample to infinite small
        hard_mask = (distance > 0).float()
        distance = distance - hard_mask * 1e9 - (1 - useful_mask) * 1e9
        # get the smallest sample which is bigger than positive sample
        semi_hard_negative_dw = torch.max(distance, dim=-1)[0]  # (batch_size, intent_tag_num, slot_tag_num)

        zero_v = torch.zeros_like(semi_hard_negative_dw).to(semi_hard_negative_dw.device)
        distance = torch.max(zero_v, semi_hard_negative_dw + self.opt.margin)
        distance = torch.mean(distance)

        return distance

    def cal_same_proto_loss(self, intent_prototype, slot_prototype):
        """push away between intent proto & slot proto"""
        intent_distance = intent_prototype.unsqueeze(-2) - intent_prototype.unsqueeze(1)
        slot_distance = slot_prototype.unsqueeze(-2) - slot_prototype.unsqueeze(1)

        intent_dw = torch.norm(intent_distance, p=2, dim=-1)  # (batch_size, intent_tag_num)
        slot_dw = torch.norm(slot_distance, p=2, dim=-1)  # (batch_size, slot_tag_num)

        intent_zero_v = torch.zeros_like(intent_dw).to(intent_dw.device)
        slot_zero_v = torch.zeros_like(slot_dw).to(slot_dw.device)

        intent_distance = torch.pow(torch.max(intent_zero_v, self.opt.margin - intent_dw), 2)
        slot_distance = torch.pow(torch.max(slot_zero_v, self.opt.margin - slot_dw), 2)

        distance = (torch.mean(intent_distance) + torch.mean(slot_distance)) / 2
        if torch.isnan(distance).sum():
            raise ValueError('there is nan in homo')
        return distance


class EmissionMergeIterationFewShotSLU(FewShotSLU):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: FewShotTextClassifier,
                 slot_decoder: FewShotSeqLabeler,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None, ):
        super(EmissionMergeIterationFewShotSLU, self).__init__(opt, context_embedder, intent_decoder, slot_decoder,
                                                               config, emb_log)
        self.intent_id2label = self.opt.id2label['intent']
        self.slot_id2label = self.opt.id2label['slot']

        if opt.context_emb == 'electra':
            emb_dim = 256
        elif opt.context_emb in ['bert', 'sep_bert', 'roberta_base']:
            emb_dim = 768
        elif opt.context_emb in ['roberta_large']:
            emb_dim = 1024
        else:
            emb_dim = opt.emb_dim

        # metric layer
        self.intent_metric = torch.nn.Linear(emb_dim, opt.metric_dim)
        self.slot_metric = torch.nn.Linear(emb_dim, opt.metric_dim)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: (batch_size, test_len)
        :param intent_test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: (batch_size, support_size, support_len)
        :param intent_support_output_mask: (batch_size, support_size, support_len)
        :param slot_test_target: index targets (batch_size, test_len)
        :param slot_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: index targets (batch_size, test_len)
        :param intent_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :return:
        """
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        ''' metric space controller '''
        sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps = self.get_reps_after_split_metric(
            sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps)

        ''' getting prototype'''
        intent_prototype, slot_prototype = None, None
        if self.opt.task in ['intent', 'slu']:
            intent_prototype = self.intent_decoder.cal_prototype(sent_test_reps, intent_test_output_mask,
                                                                 sent_support_reps, intent_support_output_mask,
                                                                 intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_prototype = self.slot_decoder.cal_prototype(seq_test_reps, slot_test_output_mask,
                                                             seq_support_reps, slot_support_output_mask,
                                                             slot_test_target, slot_support_target)

        ''' getting emission '''
        intent_emission = None
        slot_emission = None
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(sent_test_reps, intent_test_output_mask,
                                                               sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(seq_test_reps, slot_test_output_mask,
                                                           seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target)

        ''' emission merge '''
        if not self.opt.just_train_mode or self.training:
            if self.opt.emission_merge_iter_num > 0:  # merge iteration
                # check weather the task definition allow the operation
                if intent_emission is None or slot_emission is None:
                    raise ValueError('single task can not use iterative emission merge methods')
                # start merge iteration
                for i in range(self.opt.emission_merge_iter_num):
                    # expand emission as same as the one of the other task
                    expand_intent_emission = self.expand_intent_emission_to_slot(
                        slot_support_target, intent_support_target, self.slot_id2label, self.intent_id2label,
                        intent_emission)
                    expand_slot_emission = self.expand_slot_emission_to_intent(
                        slot_support_target, intent_support_target, self.slot_id2label, self.intent_id2label,
                        slot_emission)
                    # update emission with the one of the other task
                    # update method is to concat them
                    self.intent_decoder.cat_emission(expand_slot_emission)
                    self.slot_decoder.cat_emission(expand_intent_emission)
                    # get the updated emission
                    intent_emission = self.intent_decoder.get_emission()
                    slot_emission = self.slot_decoder.get_emission()
            else:  # none merge
                pass

        ''' return the result (loss / prediction) '''
        if self.training:
            loss = 0.
            if self.learning_task in ['intent', 'slu']:
                intent_loss = self.intent_decoder(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                  intent_support_output_mask, intent_test_target,
                                                  intent_support_target, self.label_mask)
                loss += intent_loss
                self.loss_dict['intent'] = intent_loss

            if self.learning_task in ['slot_filling', 'slu']:
                slot_loss = self.slot_decoder(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                              slot_support_output_mask, slot_test_target, slot_support_target,
                                              self.label_mask)
                loss += slot_loss
                self.loss_dict['slot'] = slot_loss

            if self.learning_task == 'slu':
                regular_loss = self.add_contrastive_loss(intent_support_target, slot_support_target,
                                                         intent_prototype, slot_prototype)
                loss += self.opt.slu_regular_rate * regular_loss

            return loss
        else:
            if self.opt.record_proto:
                self.record_intent_prototype = intent_prototype
                self.record_slot_prototype = slot_prototype
            intent_preds, slot_preds = None, None
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(sent_test_reps, intent_test_output_mask,
                                                          sent_support_reps,
                                                          intent_support_output_mask, intent_test_target,
                                                          intent_support_target, self.label_mask)
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(seq_test_reps, slot_test_output_mask,
                                                      seq_support_reps,
                                                      slot_support_output_mask, slot_test_target,
                                                      slot_support_target, self.label_mask,)
            return {'slot': slot_preds, 'intent': intent_preds}


class EmissionMergeIntentFewShotSLU(FewShotSLU):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: FewShotTextClassifier,
                 slot_decoder: FewShotSeqLabeler,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None, ):
        super(EmissionMergeIntentFewShotSLU, self).__init__(opt, context_embedder, intent_decoder, slot_decoder,
                                                            config, emb_log)
        self.intent_id2label = self.opt.id2label['intent']
        self.slot_id2label = self.opt.id2label['slot']

        if opt.context_emb == 'electra':
            emb_dim = 256
        elif opt.context_emb in ['bert', 'sep_bert', 'roberta_base']:
            emb_dim = 768
        elif opt.context_emb in ['roberta_large']:
            emb_dim = 1024
        else:
            emb_dim = opt.emb_dim

        # metric layer
        self.intent_metric = torch.nn.Linear(emb_dim, opt.metric_dim)
        self.slot_metric = torch.nn.Linear(emb_dim, opt.metric_dim)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: (batch_size, test_len)
        :param intent_test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: (batch_size, support_size, support_len)
        :param intent_support_output_mask: (batch_size, support_size, support_len)
        :param slot_test_target: index targets (batch_size, test_len)
        :param slot_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: index targets (batch_size, test_len)
        :param intent_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :return:
        """
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        ''' metric space controller '''
        sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps = self.get_reps_after_split_metric(
            sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps)

        ''' getting prototype'''
        intent_prototype, slot_prototype = None, None
        if self.opt.task in ['intent', 'slu']:
            intent_prototype = self.intent_decoder.cal_prototype(sent_test_reps, intent_test_output_mask,
                                                                 sent_support_reps, intent_support_output_mask,
                                                                 intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_prototype = self.slot_decoder.cal_prototype(seq_test_reps, slot_test_output_mask,
                                                             seq_support_reps, slot_support_output_mask,
                                                             slot_test_target, slot_support_target)
        ''' getting emission '''
        intent_emission = None
        slot_emission = None
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(sent_test_reps, intent_test_output_mask,
                                                               sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(seq_test_reps, slot_test_output_mask,
                                                           seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target)
        ''' merge intent emission to slot'''
        if not self.opt.just_train_mode or self.training:
            expand_intent_emission = self.expand_intent_emission_to_slot(slot_support_target, intent_support_target,
                                                                         self.slot_id2label, self.intent_id2label,
                                                                         intent_emission)
            # update emission with the one of the other task
            # update method is to concat them
            self.slot_decoder.cat_emission(expand_intent_emission)

        if self.training:
            loss = 0.
            if self.learning_task in ['intent', 'slu']:
                intent_loss = self.intent_decoder(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                  intent_support_output_mask, intent_test_target,
                                                  intent_support_target, self.label_mask)
                loss += intent_loss
                self.loss_dict['intent'] = intent_loss

            if self.learning_task in ['slot_filling', 'slu']:
                slot_loss = self.slot_decoder(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                              slot_support_output_mask,
                                              slot_test_target, slot_support_target, self.label_mask)
                loss += slot_loss
                self.loss_dict['slot'] = slot_loss

            if self.learning_task == 'slu':
                regular_loss = self.add_contrastive_loss(intent_support_target, slot_support_target,
                                                         intent_prototype, slot_prototype)

                loss += self.opt.slu_regular_rate * regular_loss

            return loss
        else:
            if self.opt.record_proto:
                self.record_intent_prototype = intent_prototype
                self.record_slot_prototype = slot_prototype
            intent_preds, slot_preds = None, None
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(sent_test_reps, intent_test_output_mask,
                                                          sent_support_reps,
                                                          intent_support_output_mask, intent_test_target,
                                                          intent_support_target, self.label_mask)
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(seq_test_reps, slot_test_output_mask,
                                                      seq_support_reps,
                                                      slot_support_output_mask, slot_test_target,
                                                      slot_support_target, self.label_mask)
            return {'slot': slot_preds, 'intent': intent_preds}


class EmissionMergeSlotFewShotSLU(FewShotSLU):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: FewShotTextClassifier,
                 slot_decoder: FewShotSeqLabeler,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None, ):
        super(EmissionMergeSlotFewShotSLU, self).__init__(opt, context_embedder, intent_decoder, slot_decoder,
                                                          config, emb_log)
        self.intent_id2label = self.opt.id2label['intent']
        self.slot_id2label = self.opt.id2label['slot']

        if opt.context_emb == 'electra':
            emb_dim = 256
        elif opt.context_emb in ['bert', 'sep_bert', 'roberta_base']:
            emb_dim = 768
        elif opt.context_emb in ['roberta_large']:
            emb_dim = 1024
        else:
            emb_dim = opt.emb_dim

        # metric layer
        self.intent_metric = torch.nn.Linear(emb_dim, opt.metric_dim)
        self.slot_metric = torch.nn.Linear(emb_dim, opt.metric_dim)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: (batch_size, test_len)
        :param intent_test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: (batch_size, support_size, support_len)
        :param intent_support_output_mask: (batch_size, support_size, support_len)
        :param slot_test_target: index targets (batch_size, test_len)
        :param slot_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: index targets (batch_size, test_len)
        :param intent_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :return:
        """
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        ''' metric space controller '''
        sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps = self.get_reps_after_split_metric(
            sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps)

        ''' getting prototype'''
        intent_prototype, slot_prototype = None, None
        if self.opt.task in ['intent', 'slu']:
            intent_prototype = self.intent_decoder.cal_prototype(sent_test_reps, intent_test_output_mask,
                                                                 sent_support_reps, intent_support_output_mask,
                                                                 intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_prototype = self.slot_decoder.cal_prototype(seq_test_reps, slot_test_output_mask,
                                                             seq_support_reps, slot_support_output_mask,
                                                             slot_test_target, slot_support_target)
        ''' getting emission '''
        intent_emission = None
        slot_emission = None
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(sent_test_reps, intent_test_output_mask,
                                                               sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(seq_test_reps, slot_test_output_mask,
                                                           seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target)

        ''' merge slot emission to intent '''
        if not self.opt.just_train_mode or self.training:
            # expand emission as same as the one of the other task
            expand_slot_emission = self.expand_slot_emission_to_intent(slot_support_target, intent_support_target,
                                                                       self.slot_id2label, self.intent_id2label,
                                                                       slot_emission)
            # update emission with the one of the other task
            # update method is to concat them
            self.intent_decoder.cat_emission(expand_slot_emission)

        if self.training:
            loss = 0.
            if self.learning_task in ['slot_filling', 'slu']:
                intent_loss = self.slot_decoder(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                                slot_support_output_mask,
                                                slot_test_target, slot_support_target, self.label_mask)
                loss += intent_loss
                self.loss_dict['intent'] = intent_loss

            if self.learning_task in ['intent', 'slu']:
                slot_loss = self.intent_decoder(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                intent_support_output_mask, intent_test_target,
                                                intent_support_target, self.label_mask)
                loss += slot_loss
                self.loss_dict['slot'] = slot_loss

            if self.learning_task == 'slu':
                regular_loss = self.add_contrastive_loss(intent_support_target, slot_support_target, intent_prototype,
                                                         slot_prototype)
                loss += self.opt.slu_regular_rate * regular_loss

            return loss
        else:
            if self.opt.record_proto:
                self.record_intent_prototype = intent_prototype
                self.record_slot_prototype = slot_prototype
            intent_preds, slot_preds = None, None
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                                      slot_support_output_mask, slot_test_target,
                                                      slot_support_target, self.label_mask)
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                          intent_support_output_mask, intent_test_target,
                                                          intent_support_target, self.label_mask)
            return {'slot': slot_preds, 'intent': intent_preds}


class ProtoMergeFewShotSLU(FewShotSLU):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: FewShotTextClassifier,
                 slot_decoder: FewShotSeqLabeler,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None, ):
        super(ProtoMergeFewShotSLU, self).__init__(opt, context_embedder, intent_decoder, slot_decoder,
                                                   config, emb_log)
        if opt.context_emb == 'electra':
            self.emb_dim = 256
        elif opt.context_emb in ['bert', 'sep_bert', 'roberta_base']:
            self.emb_dim = 768
        elif opt.context_emb in ['roberta_large']:
            self.emb_dim = 1024
        else:
            self.emb_dim = opt.emb_dim

        self.intent_id2label = self.opt.id2label['intent']
        self.slot_id2label = self.opt.id2label['slot']

        self.record_num = 0

        # metric layer
        self.intent_metric = torch.nn.Linear(self.emb_dim, opt.metric_dim)
        self.slot_metric = torch.nn.Linear(self.emb_dim, opt.metric_dim)

        # attention parameter matrix W
        if self.opt.proto_merge_type == 'add_attention':
            self.add_W = nn.Parameter(torch.empty(self.emb_dim, self.opt.attn_hidden_size))
            self.add_U = nn.Parameter(torch.empty(self.emb_dim, self.opt.attn_hidden_size))
            self.add_v = nn.Parameter(torch.empty(self.opt.attn_hidden_size, 1))
            init.xavier_normal_(self.add_W)
            init.xavier_normal_(self.add_U)
            init.xavier_normal_(self.add_v)
        elif self.opt.proto_merge_type == '2linear_attention':
            self.dlin_W = nn.Parameter(torch.empty(self.emb_dim, self.emb_dim))
            init.xavier_normal_(self.dlin_W)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
            training: bool = False,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: (batch_size, test_len)
        :param intent_test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: (batch_size, support_size, support_len)
        :param intent_support_output_mask: (batch_size, support_size, support_len)
        :param slot_test_target: index targets (batch_size, test_len)
        :param slot_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: index targets (batch_size, test_len)
        :param intent_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :param training: bool
        :return:
        """
        batch_size = test_token_ids.size(0)

        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        ''' metric space controller '''
        metric_sent_test_reps, metric_seq_test_reps, metric_sent_support_reps, metric_seq_support_reps = \
            self.get_reps_after_split_metric(sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps)

        ''' getting prototype'''
        intent_prototype, slot_prototype = None, None
        if self.opt.task in ['intent', 'slu']:
            intent_prototype = self.intent_decoder.cal_prototype(metric_sent_test_reps, intent_test_output_mask,
                                                                 metric_sent_support_reps, intent_support_output_mask,
                                                                 intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_prototype = self.slot_decoder.cal_prototype(metric_seq_test_reps, slot_test_output_mask,
                                                             metric_seq_support_reps, slot_support_output_mask,
                                                             slot_test_target, slot_support_target)

        ''' merge prototype '''
        if not self.opt.just_train_mode or self.training:
            if self.opt.task == 'slu':
                intent_num_tags = intent_prototype.size(1)
                slot_num_tags = slot_prototype.size(1)
                if self.opt.proto_merge_type == 'cat_intent_to_slot':
                    ''' merge intent to slot'''
                    if self.opt.pm_use_attn:
                        no_pad_slot_prototype_reps = self.get_slot_prototype_by_intent(slot_support_target, intent_support_target,
                                                                                       self.slot_id2label, self.intent_id2label,
                                                                                       intent_prototype)
                        slot_prototype_reps = torch.cat((slot_prototype[:, :1, :], no_pad_slot_prototype_reps), dim=1)
                    else:
                        slot_prototype_reps = self.expand_intent_to_slot_prototype(slot_support_target, intent_support_target,
                                                                                   self.slot_id2label, self.intent_id2label,
                                                                                   intent_prototype)
                    # stop the gradient update of merged prototype
                    if self.opt.pm_stop_grad:
                        with torch.no_grad():
                            slot_prototype_reps = slot_prototype_reps.detach()

                    if self.opt.proto_replace:
                        self.slot_decoder.set_prototype(slot_prototype_reps)
                    else:
                        self.slot_decoder.update_prototype(slot_prototype_reps)
                elif self.opt.proto_merge_type == 'intent_derived_by_slot':
                    ''' merge slot to intent'''
                    if self.opt.pm_use_attn:
                        no_pad_intent_prototype_reps = self.get_intent_prototype_by_slot(
                            slot_support_target, intent_support_target, self.slot_id2label, self.intent_id2label,
                            slot_prototype)
                        intent_prototype_reps = torch.cat((intent_prototype[:, :1, :], no_pad_intent_prototype_reps), dim=1)
                    else:
                        intent_prototype_reps = self.get_intent_prototype(slot_support_target, intent_support_target,
                                                                          self.slot_id2label, self.intent_id2label,
                                                                          slot_prototype)
                    # stop the gradient update of merged prototype
                    if self.opt.pm_stop_grad:
                        with torch.no_grad():
                            intent_prototype_reps = intent_prototype_reps.detach()

                    if self.opt.proto_replace:
                        self.intent_decoder.set_prototype(intent_prototype_reps)
                    else:
                        self.intent_decoder.update_prototype(intent_prototype_reps)
                elif self.opt.proto_merge_type == 'merge_both':
                    ''' merge intent to slot  & slot to intent'''
                    if self.opt.pm_use_attn:
                        no_pad_intent_prototype_reps = self.get_intent_prototype_by_slot(
                            slot_support_target, intent_support_target, self.slot_id2label, self.intent_id2label,
                            slot_prototype)
                        intent_prototype_reps = torch.cat((intent_prototype[:, :1, :], no_pad_intent_prototype_reps),
                                                          dim=1)

                        no_pad_slot_prototype_reps = self.get_slot_prototype_by_intent(
                            slot_support_target, intent_support_target, self.slot_id2label, self.intent_id2label,
                            intent_prototype)
                        slot_prototype_reps = torch.cat((slot_prototype[:, :1, :], no_pad_slot_prototype_reps), dim=1)
                    else:
                        intent_prototype_reps = self.get_intent_prototype(slot_support_target, intent_support_target,
                                                                          self.slot_id2label, self.intent_id2label,
                                                                          slot_prototype)
                        slot_prototype_reps = self.expand_intent_to_slot_prototype(
                            slot_support_target, intent_support_target, self.slot_id2label, self.intent_id2label,
                            intent_prototype)
                    # stop the gradient update of merged prototype
                    if self.opt.pm_stop_grad:
                        with torch.no_grad():
                            intent_prototype_reps = intent_prototype_reps.detach()
                            slot_prototype_reps = slot_prototype_reps.detach()

                    if self.opt.proto_replace:
                        self.intent_decoder.set_prototype(intent_prototype_reps)
                        self.slot_decoder.set_prototype(slot_prototype_reps)
                    else:
                        self.intent_decoder.update_prototype(intent_prototype_reps)
                        self.slot_decoder.update_prototype(slot_prototype_reps)
                elif self.opt.proto_merge_type == 'fix':
                    score_matrix = self.opt.intent2slot_mask.to(intent_prototype.device)
                    # get attn score
                    intent_sum = score_matrix.sum(dim=1)
                    intent_sum += (intent_sum == 0).float()
                    intent_attn = score_matrix / intent_sum.unsqueeze(-1)
                    intent_attn = intent_attn.unsqueeze(0).repeat(batch_size, 1, 1)

                    slot_sum = torch.sum(score_matrix.t(), dim=1)
                    slot_sum += (slot_sum == 0).float()
                    slot_attn = score_matrix.t() / slot_sum.unsqueeze(-1)
                    slot_attn = slot_attn.unsqueeze(0).repeat(batch_size, 1, 1)

                    # get attn_prototype
                    # (batch, intent_num_tags, emb_dim)
                    attn_intent_prototype = torch.sum(intent_attn.unsqueeze(-1) * slot_prototype.unsqueeze(1), dim=2)
                    # (batch, slot_num_tags, emb_dim)
                    attn_slot_prototype = torch.sum(slot_attn.unsqueeze(-1) * intent_prototype.unsqueeze(1), dim=2)

                    # update operation has more controllable params: pr_nm, pr_scl, proto_scale_r
                    if self.opt.proto_update in ['intent', 'slu']:
                        # update intent with slot
                        self.intent_decoder.update_prototype(attn_intent_prototype)
                    if self.opt.proto_update in ['slot', 'slu']:
                        # update intent with intent
                        self.slot_decoder.update_prototype(attn_slot_prototype)
                elif self.opt.proto_merge_type in ['dot_attention', 'scale_dot_attention', 'add_attention', '2linear_attention']:
                    # dot product attention model
                    # (batch, intent_num_tags, slot_num_tags)
                    if self.opt.proto_merge_type == 'add_attention':
                        wx = torch.bmm(intent_prototype, self.add_W.unsqueeze(0).repeat(batch_size, 1, 1)).unsqueeze(-2)  # (b, i_n, 1, h)
                        uq = torch.bmm(slot_prototype, self.add_U.unsqueeze(0).repeat(batch_size, 1, 1)).unsqueeze(-3)  # (b, 1, s_n, h)
                        wx_uq = (wx + uq).contiguous().view(batch_size, -1, self.opt.attn_hidden_size)  # (b, i_n * s_n, h)
                        score_matrix = torch.bmm(wx_uq, self.add_v.unsqueeze(0).repeat(batch_size, 1, 1))  # (b, i_n * s_n, 1)
                        score_matrix = score_matrix.view(batch_size, intent_num_tags, slot_num_tags)
                    elif self.opt.proto_merge_type == '2linear_attention':
                        score_matrix = torch.bmm(
                            torch.bmm(intent_prototype, self.dlin_W.unsqueeze(0).repeat(batch_size, 1, 1)),
                            slot_prototype.transpose(1, 2))
                    else:
                        score_matrix = torch.bmm(intent_prototype, slot_prototype.transpose(2, 1))
                        if self.opt.proto_merge_type == 'scale_dot_attention':
                            score_matrix = torch.div(score_matrix, self.emb_dim ** 0.5)  # divide the sqrt vector dimension

                    if self.opt.attn_spt_mask:
                        intent2slot_mask = self.get_support_intent2slot_mask(
                            intent_support_target, slot_support_target, self.intent_id2label, self.slot_id2label)
                        intent2slot_mask = intent2slot_mask.to(test_token_ids.device)
                        score_matrix = score_matrix.masked_fill(intent2slot_mask == 0, -1e9)
                    elif self.opt.attn_glb_mask:
                        intent2slot_mask = self.opt.intent2slot_mask.to(test_token_ids.device)
                        intent2slot_mask = intent2slot_mask.unsqueeze(0).repeat(batch_size, 1, 1)
                        score_matrix = score_matrix.masked_fill(intent2slot_mask == 0, -1e9)

                    # get attn score
                    intent_attn = torch.softmax(score_matrix, dim=2)  # intent as query (batch, intent_num_tags, score_dim)
                    slot_attn = torch.softmax(score_matrix, dim=1).permute(0, 2, 1)  # slot as query (batch, slot_num_tags, score_dim)
                    # get attn_prototype
                    # (batch, intent_num_tags, emb_dim)
                    attn_intent_prototype = torch.sum(intent_attn.unsqueeze(-1) * slot_prototype.unsqueeze(1), dim=2)
                    # (batch, slot_num_tags, emb_dim)
                    attn_slot_prototype = torch.sum(slot_attn.unsqueeze(-1) * intent_prototype.unsqueeze(1), dim=2)

                    # stop the gradient update of merged prototype
                    if self.opt.pm_stop_grad:
                        with torch.no_grad():
                            attn_intent_prototype = attn_intent_prototype.detach()
                            attn_slot_prototype = attn_slot_prototype.detach()

                    if self.opt.pm_use_attn:
                        no_pad_intent_prototype_reps = self.get_intent_prototype_by_slot(
                            slot_support_target, intent_support_target, self.slot_id2label, self.intent_id2label,
                            slot_prototype)
                        intent_prototype_reps = torch.cat((intent_prototype[:, :1, :], no_pad_intent_prototype_reps),
                                                          dim=1)

                        no_pad_slot_prototype_reps = self.get_slot_prototype_by_intent(
                            slot_support_target, intent_support_target, self.slot_id2label, self.intent_id2label,
                            intent_prototype)
                        slot_prototype_reps = torch.cat((slot_prototype[:, :1, :], no_pad_slot_prototype_reps), dim=1)

                        attn_intent_prototype = self.opt.pm_attn_r * intent_prototype_reps + (1 - self.opt.pm_attn_r) * attn_intent_prototype
                        attn_slot_prototype = self.opt.pm_attn_r * slot_prototype_reps + (1 - self.opt.pm_attn_r) * attn_slot_prototype

                    # update operation has more controllable params: pr_nm, pr_scl, proto_scale_r
                    if self.opt.proto_update in ['intent', 'slu']:
                        # update intent with slot
                        self.intent_decoder.update_prototype(attn_intent_prototype)
                    if self.opt.proto_update in ['slot', 'slu']:
                        # update intent with intent
                        self.slot_decoder.update_prototype(attn_slot_prototype)
                elif self.opt.proto_merge_type == 'logit_attention':
                    pass
                else:
                    raise NotImplementedError

        ''' get new prototype '''
        intent_prototype = self.intent_decoder.get_prototype()
        slot_prototype = self.slot_decoder.get_prototype()

        ''' getting emission '''
        intent_emission, slot_emission = None, None
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(metric_sent_test_reps, intent_test_output_mask,
                                                               metric_sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(metric_seq_test_reps, slot_test_output_mask,
                                                           metric_seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target)

        ''' emission merge '''
        if not self.opt.just_train_mode or self.training:
            if self.opt.emission_merge_type == 'iteration':
                if self.opt.emission_merge_iter_num > 0:  # merge iteration
                    # check weather the task definition allow the operation
                    if intent_emission is None or slot_emission is None:
                        raise ValueError('single task can not use iterative emission merge methods')
                    # start merge iteration
                    for i in range(self.opt.emission_merge_iter_num):
                        # expand emission as same as the one of the other task
                        expand_intent_emission = self.expand_intent_emission_to_slot(slot_support_target, intent_support_target,
                                                                                     self.slot_id2label, self.intent_id2label,
                                                                                     intent_emission)
                        expand_slot_emission = self.expand_slot_emission_to_intent(slot_support_target, intent_support_target,
                                                                                   self.slot_id2label, self.intent_id2label,
                                                                                   slot_emission)
                        # update emission with the one of the other task
                        # update method is to concat them
                        self.intent_decoder.cat_emission(expand_slot_emission)
                        self.slot_decoder.cat_emission(expand_intent_emission)
                        # get the updated emission
                        intent_emission = self.intent_decoder.get_emission()
                        slot_emission = self.slot_decoder.get_emission()
                else:  # none merge
                    pass
            elif self.opt.emission_merge_type == 'intent':
                # merge intent emission to slot
                expand_intent_emission = self.expand_intent_emission_to_slot(slot_support_target, intent_support_target,
                                                                             self.slot_id2label, self.intent_id2label,
                                                                             intent_emission)
                # update emission with the one of the other task
                # update method is to concat them
                self.slot_decoder.cat_emission(expand_intent_emission)
            elif self.opt.emission_merge_type == 'slot':
                # merge slot emission to intent
                # expand emission as same as the one of the other task
                expand_slot_emission = self.expand_slot_emission_to_intent(slot_support_target, intent_support_target,
                                                                           self.slot_id2label, self.intent_id2label,
                                                                           slot_emission)
                # update emission with the one of the other task
                # update method is to concat them
                self.intent_decoder.cat_emission(expand_slot_emission)
            elif self.opt.emission_merge_type == 'none':
                pass
            else:
                raise NotImplementedError

        ''' use emission logit to update proto '''
        if not self.opt.just_train_mode or self.training:
            if self.opt.task == 'slu' and self.opt.proto_merge_type == 'logit_attention':
                # add pad back to emission, and sum all token emission in slot
                pad_intent_emission = torch.cat((torch.zeros(batch_size, 1, 1).to(test_token_ids.device), intent_emission), dim=-1)
                pad_slot_emission = torch.cat((torch.zeros(batch_size, 1, 1).to(test_token_ids.device),
                                               torch.sum(slot_emission, dim=-2, keepdim=True)), dim=-1)
                # get intent2slot realtion mask
                intent2slot_mask = self.get_support_intent2slot_mask(intent_support_target, slot_support_target,
                                                                     self.intent_id2label, self.slot_id2label)
                intent2slot_mask = intent2slot_mask.to(test_token_ids.device)

                ''' start use emission logit to merge prototype '''
                if self.learning_task in ['intent', 'slu']:
                    # cal intent emission logits * intent prototype, add it to slot prototype
                    slot2intent_mask = intent2slot_mask.transpose(-1, -2)  # (batch_size, slot_num_tags, intent_num_tags)
                    intent_attn = torch.softmax(slot2intent_mask * pad_intent_emission, dim=-1)
                    # (batch_size, slot_num_tags, emb_dim)
                    intent_attn_proto = torch.sum(intent_attn.unsqueeze(-1) * intent_prototype.unsqueeze(1), dim=-2)
                    self.slot_decoder.update_prototype(intent_attn_proto)

                if self.learning_task in ['slot_filling', 'slu']:
                    # cal slot emission logits * slot prototype, add it to intent prototype
                    # (batch_size, intent_num_tags, slot_num_tags)
                    slot_attn = torch.softmax(intent2slot_mask * pad_slot_emission, dim=-1)
                    # (batch_size, intent_num_tags, emb_dim)
                    slot_attn_proto = torch.sum(slot_attn.unsqueeze(-1) * slot_prototype.unsqueeze(1), dim=-2)
                    self.intent_decoder.update_prototype(slot_attn_proto)

        ''' cal loss '''
        loss = 0.
        if self.learning_task in ['intent', 'slu']:
            loss += self.intent_decoder(metric_sent_test_reps, intent_test_output_mask, metric_sent_support_reps,
                                        intent_support_output_mask, intent_test_target,
                                        intent_support_target, self.label_mask)
        if self.learning_task in ['slot_filling', 'slu']:
            slot_loss = self.slot_decoder(metric_seq_test_reps, slot_test_output_mask, metric_seq_support_reps,
                                          slot_support_output_mask,
                                          slot_test_target, slot_support_target, self.label_mask)
            loss += self.opt.loss_rate * slot_loss
        if self.learning_task == 'slu':
            regular_loss = self.add_contrastive_loss(intent_support_target, slot_support_target, intent_prototype,
                                                     slot_prototype)
            loss += self.opt.slu_regular_rate * regular_loss
        
        if training:
            intent_preds, slot_preds = None, None
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.intent_emission
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.slot_emission
            return loss, (intent_preds, slot_preds)
        
        ''' training '''
        if self.training:
            return loss
        else:
            '''store visualization embedding'''
            if self.opt.record_proto:
                self.record_intent_prototype = intent_prototype.detach().cpu()
                self.record_slot_prototype = slot_prototype.detach().cpu()

            intent_preds, slot_preds = None, None
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(metric_sent_test_reps, intent_test_output_mask,
                                                          metric_sent_support_reps, intent_support_output_mask,
                                                          intent_test_target, intent_support_target, self.label_mask)
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(metric_seq_test_reps, slot_test_output_mask,
                                                      metric_seq_support_reps, slot_support_output_mask,
                                                      slot_test_target, slot_support_target, self.label_mask)
            return {'slot': slot_preds, 'intent': intent_preds}

    def expand_intent_to_slot_prototype(self,
                                        slot_support_target,
                                        intent_support_target,
                                        slot_id2label,
                                        intent_id2label,
                                        intent_prototype) -> torch.Tensor:
        """
        get intent prototype reps by deriving from slot prototype reps, via support set statistic
        :param slot_support_target: (batch_size, support_size, test_len)
        :param intent_support_target: (batch_size, support_size, 1)
        :param slot_id2label:
        :param intent_id2label:
        :param intent_prototype: (batch_size, intent_num_tags, dim)
        :return: (batch_size, intent_num_tags, dim)
        """
        slot_support_target = self.onehot2label_id(slot_support_target)
        intent_support_target = self.onehot2label_id(intent_support_target)

        batch_size, support_size, test_len = slot_support_target.size()
        dim = intent_prototype.size()[-1]

        intent_support_target = intent_support_target.view(batch_size, -1).tolist()

        batch_expand_intent_prototype = []
        for b_idx in range(batch_size):
            slot2intent_lst = {slot_id: Counter() for slot_id in slot_id2label}
            for s_idx in range(support_size):
                intent_label_id = intent_support_target[b_idx][s_idx]
                for slot_label_id in slot_support_target[b_idx][s_idx]:
                    if slot_id2label[slot_label_id.item()] not in ['O', '[PAD]']:
                        slot2intent_lst[slot_label_id.item()][intent_label_id] += 1

            expand_intent_prototype = []
            for slot_id in slot_id2label:
                if len(slot2intent_lst[slot_id]) == 0:
                    expand_intent_prototype.append(torch.zeros(dim).to(intent_prototype.device))
                else:
                    all_intent_label_ids = list(slot2intent_lst[slot_id].keys())
                    expand_intent_prototype.append(torch.mean(intent_prototype[b_idx][all_intent_label_ids], dim=0))
            expand_intent_prototype = torch.stack(expand_intent_prototype, dim=0)
            batch_expand_intent_prototype.append(expand_intent_prototype)

        batch_expand_intent_prototype = torch.stack(batch_expand_intent_prototype, dim=0)
        return batch_expand_intent_prototype

    def get_slot_prototype_by_intent(self,
                                     slot_support_target,
                                     intent_support_target,
                                     slot_id2label,
                                     intent_id2label,
                                     intent_prototype) -> torch.Tensor:
        """
        get slot prototype reps by deriving from intent prototype reps, via support set statistic
        :param slot_support_target: (batch_size, support_size, test_len)
        :param intent_support_target: (batch_size, support_size, 1)
        :param slot_id2label:
        :param intent_id2label:
        :param intent_prototype: (batch_size, intent_num_tags, dim)
        :return: (batch_size, intent_num_tags, dim)
        """
        batch_intent2slot_map = self.cal_intent2slot_map_from_support(
            intent_support_target, slot_support_target, intent_id2label, slot_id2label)

        no_pad_batch_intent2slot_map = batch_intent2slot_map[:, 1:, 1:] + 1  # del [PAD] & softplus
        # intent as query, get attention for slot (batch_size, intent_num_tags - 1, slot_num_tag - 1)
        batch_slot_query_attn = torch.softmax(no_pad_batch_intent2slot_map, dim=-2).transpose(1, 2).to(intent_prototype.device)

        no_pad_intent_prototype = intent_prototype[:, 1:, :]
        no_pad_slot_prototype = torch.sum(batch_slot_query_attn.unsqueeze(-1) * no_pad_intent_prototype.unsqueeze(1),
                                          dim=2)
        return no_pad_slot_prototype

    def get_intent_prototype(self,
                             slot_support_target,
                             intent_support_target,
                             slot_id2label,
                             intent_id2label,
                             slot_prototype) -> torch.Tensor:
        """
        get intent prototype reps by deriving from slot prototype reps, via support set statistic
        :param slot_support_target: (batch_size, support_size, test_len)
        :param intent_support_target: (batch_size, support_size, 1)
        :param slot_id2label:
        :param intent_id2label:
        :param slot_prototype: (batch_size, slot_num_tags, dim)
        :return: (batch_size, intent_num_tags, dim)
        """
        slot_support_target = self.onehot2label_id(slot_support_target)
        intent_support_target = self.onehot2label_id(intent_support_target)

        batch_size, support_size, test_len = slot_support_target.size()
        dim = slot_prototype.size()[-1]

        intent_support_target = intent_support_target.view(batch_size, -1).tolist()

        intent_prototype = []
        for b_idx in range(batch_size):
            tmp_slots = {}
            tmp_intent_reps = {intent_id: [] for intent_id in intent_id2label}
            for s_idx in range(support_size):
                intent_label_id = intent_support_target[b_idx][s_idx]
                if intent_label_id not in tmp_slots:
                    tmp_slots[intent_label_id] = []
                tmp_slots[intent_label_id].extend(slot_support_target[b_idx][s_idx].tolist())
            for label_id, slots in tmp_slots.items():
                slots = list(set(slots))
                for slot_id in slots:
                    tmp_intent_reps[label_id].append(slot_prototype[b_idx][slot_id])
            tmp_reps = []
            for intent_id in intent_id2label:
                if len(tmp_intent_reps[intent_id]) == 0:
                    tmp_reps.append(torch.zeros(dim).to(slot_prototype.device))
                else:
                    intent_id_lst = torch.stack(tmp_intent_reps[intent_id], dim=0).to(slot_prototype.device)
                    tmp_reps.append(torch.mean(intent_id_lst, dim=0))
            intent_prototype.append(torch.stack(tmp_reps, dim=0).to(slot_prototype.device))

        intent_prototype = torch.stack(intent_prototype, dim=0).to(slot_prototype.device)
        return intent_prototype

    def get_intent_prototype_by_slot(self,
                                     slot_support_target,
                                     intent_support_target,
                                     slot_id2label,
                                     intent_id2label,
                                     slot_prototype) -> torch.Tensor:
        """
        get intent prototype reps by deriving from slot prototype reps, via support set statistic
        :param slot_support_target: (batch_size, support_size, test_len)
        :param intent_support_target: (batch_size, support_size, 1)
        :param slot_id2label:
        :param intent_id2label:
        :param slot_prototype: (batch_size, slot_num_tags, dim)
        :return: (batch_size, intent_num_tags, dim)
        """
        batch_intent2slot_map = self.cal_intent2slot_map_from_support(
            intent_support_target, slot_support_target, intent_id2label, slot_id2label)

        no_pad_batch_intent2slot_map = batch_intent2slot_map[:, 1:, 1:] + 1  # del [PAD] & softplus
        # intent as query, get attention for slot (batch_size, intent_num_tags - 1, slot_num_tag - 1)
        batch_intent_query_attn = torch.softmax(no_pad_batch_intent2slot_map, dim=-1).to(slot_prototype.device)

        no_pad_slot_prototype = slot_prototype[:, 1:, :]

        no_pad_intent_prototype = torch.sum(batch_intent_query_attn.unsqueeze(-1) * no_pad_slot_prototype.unsqueeze(1),
                                            dim=2)
        return no_pad_intent_prototype


class SchemaFewShotSLU(FewShotSLU):
    def __init__(
            self,
            opt,
            context_embedder: ContextEmbedderBase,
            intent_decoder: FewShotTextClassifier,
            slot_decoder: FewShotSeqLabeler,
            config: dict = None,  # store necessary setting or none-torch params
            emb_log: str = None
    ):
        super(SchemaFewShotSLU, self).__init__(opt, context_embedder, intent_decoder, slot_decoder, config, emb_log)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
            slot_label_input: Tuple[torch.Tensor] = None,
            intent_label_input: Tuple[torch.Tensor] = None,
    ):
        """
        few-shot sequence labeler using schema information
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: A dict of (batch_size, test_len)
        :param intent_test_output_mask: A dict of (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: A dict of (batch_size, support_size, support_len)
        :param intent_support_output_mask: A dict of (batch_size, support_size, support_len)
        :param slot_test_target: A dict of index targets (batch_size, test_len)
        :param slot_support_target: A dict of one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: A dict of index targets (batch_size, test_len)
        :param intent_support_target: A dict of one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :param slot_label_input: include
            label_token_ids: A tensor which is same to label token ids
                if label_reps=cat:
                    (batch_size, label_num * label_des_len)
                elif:
                    (batch_size, label_num, label_des_len)
            label_segment_ids: A tensor which is same to test token ids
            label_nwp_index: A tensor which is same to test token ids
            label_input_mask: A tensor which is same to label token ids
            label_output_mask: A tensor which is same to label token ids
        :param intent_label_input: include
        :return:
        """
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids,
            support_segment_ids, support_nwp_index, support_input_mask, self.opt.use_cls
        )

        slot_label_token_ids, slot_label_segment_ids, slot_label_nwp_index, slot_label_input_mask, _ = slot_label_input
        slot_label_reps = self.get_label_reps(slot_label_token_ids, slot_label_segment_ids, slot_label_nwp_index,
                                              slot_label_input_mask)
        intent_label_token_ids, intent_label_segment_ids, intent_label_nwp_index, intent_label_input_mask, _ = \
            intent_label_input
        intent_label_reps = self.get_label_reps(intent_label_token_ids, intent_label_segment_ids,
                                                intent_label_nwp_index, intent_label_input_mask)

        ''' get sentence level representation '''
        sent_test_reps, sent_support_reps = self.get_sentence_level_reps(sent_test_reps, sent_support_reps,
                                                                         seq_test_reps, seq_support_reps,
                                                                         slot_test_target, slot_support_target)

        ''' metric space controller '''
        sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps = self.get_reps_after_split_metric(
            sent_test_reps, seq_test_reps, sent_support_reps, seq_support_reps)

        ''' getting prototype'''
        intent_prototype, slot_prototype = None, None
        if self.opt.task in ['intent', 'slu']:
            intent_prototype = self.intent_decoder.cal_prototype(sent_test_reps, intent_test_output_mask,
                                                                 sent_support_reps, intent_support_output_mask,
                                                                 intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_prototype = self.slot_decoder.cal_prototype(seq_test_reps, slot_test_output_mask,
                                                             seq_support_reps, slot_support_output_mask,
                                                             slot_test_target, slot_support_target)
        ''' getting emission '''
        intent_emission, slot_emission = None, None
        intent_prototype, slot_prototype = None, None
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(sent_test_reps, intent_test_output_mask,
                                                               sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target,
                                                               intent_label_reps)
            intent_prototype = self.intent_decoder.get_prototype()  # (batch_size, num_tags, emd_dim)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(seq_test_reps, slot_test_output_mask,
                                                           seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target,
                                                           slot_label_reps)
            slot_prototype = self.slot_decoder.get_prototype()  # (batch_size, num_tags, emd_dim)

        if self.training:
            loss = 0.
            if self.learning_task in ['intent', 'slu']:
                intent_loss = self.intent_decoder(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                  intent_support_output_mask, intent_test_target,
                                                  intent_support_target, intent_label_reps, self.label_mask)
                loss += intent_loss
                self.loss_dict['intent'] = intent_loss
            if self.learning_task in ['slot_filling', 'slu']:
                slot_loss = self.slot_decoder(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                              slot_support_output_mask, slot_test_target, slot_support_target,
                                              slot_label_reps, self.label_mask)
                loss += self.opt.loss_rate * slot_loss
                self.loss_dict['slot'] = slot_loss

            regular_loss = self.add_contrastive_loss(intent_support_target, slot_support_target, intent_prototype,
                                                     slot_prototype)
            loss += self.opt.slu_regular_rate * regular_loss
            return loss
        else:
            if self.opt.record_proto:
                self.record_intent_prototype = intent_prototype.detach().cpu()
                self.record_slot_prototype = slot_prototype.detach().cpu()
            intent_preds, slot_preds = None, None
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                          intent_support_output_mask, intent_test_target,
                                                          intent_support_target, intent_label_reps, self.label_mask)
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                                      slot_support_output_mask, slot_test_target,
                                                      slot_support_target, slot_label_reps, self.label_mask)
            return {'slot': slot_preds, 'intent': intent_preds}

    def get_label_reps(
            self,
            label_token_ids: torch.Tensor,
            label_segment_ids: torch.Tensor,
            label_nwp_index: torch.Tensor,
            label_input_mask: torch.Tensor,
            use_cls: bool = False
    ) -> torch.Tensor:
        """
        :param label_token_ids:
        :param label_segment_ids:
        :param label_nwp_index:
        :param label_input_mask:
        :param use_cls:
        :return:  shape (batch_size, label_num, label_des_len)
        """
        return self.context_embedder(
            label_token_ids, label_segment_ids, label_nwp_index, label_input_mask,  reps_type='label', use_cls=use_cls)

