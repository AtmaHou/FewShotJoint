#!/usr/bin/env python3
# -*- coding: utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict


class AttentionRNNSLU(nn.Module):

    def __init__(self, opt, context_embedder, intent_size, slot_size, bos_token_id=0, config=None, emb_log=None):
        super(AttentionRNNSLU, self).__init__()
        self.opt = opt
        self.context_embedder = context_embedder
        self.config = config
        self.emb_log = emb_log
        self.bos_token_id = slot_size

        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt, intent_size, slot_size)
        self.slot_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # the [PAD] label is 0
        self.intent_loss_fn = nn.CrossEntropyLoss()

    def get_context_reps(self, token_ids, segment_ids, nwp_index, input_mask, use_cls=False):
        self.context_embedder.eval()  # to avoid the dropout effect of reps model
        self.context_embedder.requires_grad = False
        seq_reps, _ = self.context_embedder(token_ids, segment_ids, nwp_index, input_mask, use_cls=use_cls)
        seq_reps = seq_reps.detach()
        return seq_reps

    def forward(self, token_ids, segment_ids, nwp_index, input_mask, slot_output_mask, intent_output_mask,
                slot_target, intent_target, params=None):

        seq_reps = self.get_context_reps(token_ids, segment_ids, nwp_index, input_mask)

        encoder_outputs, last_real_context_hidden = self.encoder(seq_reps, nwp_index)
        batch_size = token_ids.size(0)
        start_decode_input = torch.LongTensor([[self.bos_token_id] * batch_size]).transpose(1, 0).to(token_ids.device)

        slot_scores, intent_score = self.decoder(start_decode_input, last_real_context_hidden, encoder_outputs,
                                                 slot_output_mask)

        if self.training:
            seq_len = slot_target.size(1)

            slot_loss = self.slot_loss_fn(slot_scores.view(batch_size * seq_len, -1), slot_target.view(-1))
            intent_loss = self.intent_loss_fn(intent_score, intent_target.view(-1))
            loss = self.opt.alpha * intent_loss + (1 - self.opt.alpha) * slot_loss
            return loss
        else:
            preds = {'slot': self.decode(slot_scores, slot_output_mask),
                     'intent': self.decode(intent_score.unsqueeze(1), intent_output_mask)}
            return preds

    def decode(self, logits: torch.Tensor, masks: torch.Tensor) -> List[List[int]]:
        return self.remove_pad(preds=logits.argmax(dim=-1), masks=masks)

    def remove_pad(self, preds: torch.Tensor, masks: torch.Tensor) -> List[List[int]]:
        # remove predict result for padded token
        ret = []
        for pred, mask in zip(preds, masks):
            if pred.dim() >= 1:
                temp = []
                for l_id, mk in zip(pred, mask):
                    if mk:
                        temp.append(int(l_id))
                if temp:
                    ret.append(temp)
            else:
                if mask:
                    ret.append(int(pred))
        return ret


class Encoder(nn.Module):

    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.emb_dim = self.get_emb_dim()
        self.hidden_size = opt.lstm_hidden_size
        self.n_layers = opt.n_layers
        self.bi_direction = opt.bi_direction

        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bidirectional=self.bi_direction, batch_first=True)

    def get_emb_dim(self):
        if self.opt.context_emb == 'electra':
            emb_dim = 256
        elif self.opt.context_emb in ['bert', 'sep_bert']:
            emb_dim = 768
        else:
            emb_dim = self.opt.emb_dim
        return emb_dim

    def init_hidden(self, input):
        is_double = 2 if self.bi_direction else 1
        batch_size = input.size(0)
        h0 = nn.Parameter(torch.zeros(self.n_layers * is_double, batch_size, self.hidden_size)).to(input.device)
        c0 = nn.Parameter(torch.zeros(self.n_layers * is_double, batch_size, self.hidden_size)).to(input.device)
        return h0, c0

    def forward(self, seq_reps, nwp_index, params=None):
        h0, c0 = self.init_hidden(seq_reps)
        out, _ = self.lstm(seq_reps, (h0, c0))

        real_context = []
        flatten_nwp_index = nwp_index.squeeze(-1)
        for idx, o in enumerate(out):  # (B,T,D)
            real_last_idx = (flatten_nwp_index[idx] != 0).sum().item()  # get real length of a sentence
            real_context.append(o[real_last_idx])  # get the ending word (except [PAD])

        last_real_context_hidden = torch.stack(real_context, dim=0)

        return out, last_real_context_hidden


class Decoder(nn.Module):

    def __init__(self, opt, intent_size, slot_size):
        super(Decoder, self).__init__()
        self.opt = opt
        self.intent_size = intent_size
        self.slot_size = slot_size
        self.emb_dim = slot_size // 3  # self.get_emb_dim()
        self.n_layers = opt.n_layers
        self.bi_direction = opt.bi_direction
        num_directions = 2 if self.bi_direction else 1
        self.hidden_size = opt.lstm_hidden_size * num_directions

        # Define the layers
        vocab_size = slot_size + 1  # add [BOS]
        self.embedding = nn.Embedding(vocab_size, self.emb_dim)

        self.lstm = nn.LSTM(input_size=self.emb_dim + self.hidden_size * 2,
                            hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)  # Attention
        self.slot_out = nn.Linear(self.hidden_size * 2, self.slot_size)
        self.intent_out = nn.Linear(self.hidden_size * 2, self.intent_size)

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        pass

    def get_emb_dim(self):
        if self.opt.context_emb == 'electra':
            emb_dim = 256
        elif self.opt.context_emb in ['bert', 'sep_bert']:
            emb_dim = 768
        else:
            emb_dim = self.opt.emb_dim
        return emb_dim

    def Attention(self, hidden, encoder_outputs, encoder_maskings, params=None):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """

        hidden = hidden.squeeze(0).unsqueeze(2)

        batch_size = encoder_outputs.size(0)  # B
        max_len = encoder_outputs.size(1)  # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))  # B*T,D -> B*T,D
        energies = energies.view(batch_size, max_len, -1)  # B,T,D
        attn_energies = energies.bmm(hidden).transpose(1, 2)  # B,T,D * B,D,1 --> B,1,T
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings.bool(), -1e12)  # PAD masking

        alpha = F.softmax(attn_energies, -1)  # B,T
        alpha = alpha.unsqueeze(1)  # B,1,T
        context = alpha.bmm(encoder_outputs)  # B,1,T * B,T,D => B,1,D

        return context  # B,1,D

    def init_hidden(self, input):
        batch_size = input.size(0)
        h0 = nn.Parameter(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to(input.device)
        c0 = nn.Parameter(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to(input.device)
        return h0, c0

    def extra_non_word_piece_reps(self, input_mask, nwp_index):
        flatten_nwp_index = nwp_index.squeeze(-1)
        nwp_input_mask = torch.gather(input_mask, dim=-1, index=flatten_nwp_index)
        shape = list(flatten_nwp_index.shape)
        no_head_mask = (flatten_nwp_index.narrow(-1, 1, shape[-1] - 1) != 0).long()
        head_shape = shape[:]
        head_shape[-1] = 1
        head_mask = torch.ones(head_shape).long().to(input_mask.device)
        mask = torch.cat((head_mask, no_head_mask), dim=-1)
        assert mask.size() == nwp_input_mask.size(), \
            "the mask size `{}` should equal to nwp_input_mask size `{}`".format(mask.size(), nwp_input_mask.size())
        nwp_input_mask = nwp_input_mask * mask
        return nwp_input_mask

    def forward(self, start_input, last_encoder_hidden, encoder_outputs, nwp_input_mask, params=None):
        batch_size, seq_len, dim = encoder_outputs.size()

        if self.opt.do_debug:
            print('start_input: {}'.format(start_input.size()))
            print('last_encoder_hidden: {}'.format(last_encoder_hidden.size()))
            print('encoder_outputs: {}'.format(encoder_outputs.size()))
            print('nwp_input_mask: {}'.format(nwp_input_mask.size()))
            print('nwp_input_mask: {}'.format(nwp_input_mask.tolist()))

        embedded = self.embedding(start_input)
        hidden = self.init_hidden(start_input)

        if self.opt.do_debug:
            print('embedded: {}'.format(embedded.size()))

        decode = []
        aligns = encoder_outputs.transpose(0, 1)
        intent_score = 0
        context = last_encoder_hidden.unsqueeze(1)
        res = []
        for i in range(seq_len):
            aligned = aligns[i].unsqueeze(1)  # B,1,D
            # input, context, aligned encoder hidden, hidden
            _, hidden = self.lstm(torch.cat((embedded, context, aligned), -1), hidden)

            # for Intent Detection
            if i == 0:
                intent_hidden = hidden[0].clone()
                intent_context = self.Attention(intent_hidden, encoder_outputs, nwp_input_mask)
                concated = torch.cat((intent_hidden, intent_context.transpose(0, 1)), 2)  # 1,B,D
                intent_score = self.intent_out(concated.squeeze(0))  # B,D

            concated = torch.cat((hidden[0], context.transpose(0, 1)), 2)
            score = self.slot_out(concated.squeeze(0))
            softmaxed = F.log_softmax(score, dim=-1)
            decode.append(softmaxed)
            next_input = torch.argmax(softmaxed, 1)
            res.append(next_input.tolist())
            embedded = self.embedding(next_input.unsqueeze(1))

            if self.opt.do_debug:
                print('hidden[0]: {}'.format(hidden[0].size()))
            context = self.Attention(hidden[0], encoder_outputs, nwp_input_mask)
        slot_scores = torch.stack(decode, 1)  # (B, T, D)
        if self.opt.do_debug:
            print('slot_scores: {}'.format(slot_scores.size()))
            print('res: {}'.format(res))
            print('slot_scores: {}'.format(slot_scores.tolist()))
            print('intent_score: {}'.format(intent_score.tolist()))
        return slot_scores, intent_score
