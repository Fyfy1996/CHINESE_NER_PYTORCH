# -*- coding: utf-8 -*-

import torch
from torch import nn #, optim
# from torch.utils.data import DataLoader, Dataset
from torch.nn import init
# import numpy as np
# from tensorboardX import SummaryWriter
from transformers import BertModel


def log_sum_exp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    Compute logsumexp in a numerically stable way.
    This is mathematically equivalent to ``tensor.exp().sum(dim, keep=keepdim).log()``.
    This function is typically used for summing log probabilities.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

class CRFLayer(nn.Module):
    def __init__(self, tag_size, tag2idx, START_TAG, END_TAG):
        super(CRFLayer, self).__init__()
        # transition[i][j] means transition probability from j to i
        self.transition = nn.Parameter(torch.randn(tag_size, tag_size))
        self.tag2idx = tag2idx
        self.START_TAG = START_TAG
        self.END_TAG = END_TAG
        self.reset_parameters() 
        

    def reset_parameters(self):
        init.normal_(self.transition)
        # initialize START_TAG, END_TAG probability in log space
        self.transition.detach()[self.tag2idx[self.START_TAG], :] = -10000
        self.transition.detach()[:, self.tag2idx[self.END_TAG]] = -10000

    def forward(self, feats, mask):
        """
        Arg:
            feats: (seq_len, batch_size, tag_size)
            mask: (seq_len, batch_size)
        Return:
            scores: (batch_size, )
        """
        seq_len, batch_size, tag_size = feats.size()
        # initialize alpha to zero in log space
        alpha = feats.new_full((batch_size, tag_size), fill_value=-10000)
        # alpha in START_TAG is 1
        alpha[:, self.tag2idx[self.START_TAG]] = 0
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            # emit_score is the same regardless of current_tag, so we broadcast along current_tag
            emit_score = feat.unsqueeze(-1) # (batch_size, tag_size, 1)
            # transition_score is the same regardless of each sample, so we broadcast along batch_size dimension
            transition_score = self.transition.unsqueeze(0) # (1, tag_size, tag_size)
            # alpha_score is the same regardless of next_tag, so we broadcast along next_tag dimension
            alpha_score = alpha.unsqueeze(1) # (batch_size, 1, tag_size)
            alpha_score = alpha_score + transition_score + emit_score
            # log_sum_exp along current_tag dimension to get next_tag alpha
            mask_t = mask[t].unsqueeze(-1)
            alpha = log_sum_exp(alpha_score, -1) * mask_t + alpha * (1 - mask_t) # (batch_size, tag_size)
        # arrive at END_TAG
        alpha = alpha + self.transition[self.tag2idx[self.END_TAG]].unsqueeze(0)
        return log_sum_exp(alpha, -1) # (batch_size, )

    def score_sentence(self, feats, tags, mask):
        """
        Arg:
            feats: (seq_len, batch_size, tag_size)
            tags: (seq_len, batch_size)
            mask: (seq_len, batch_size)
        Return:
            scores: (batch_size, )
        """
        seq_len, batch_size, tag_size = feats.size()
        scores = feats.new_zeros(batch_size)
        tags = torch.cat([tags.new_full((1, batch_size), fill_value=self.tag2idx[self.START_TAG]), tags], 0) # (seq_len + 1, batch_size)
        for t, feat in enumerate(feats):
            emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[t + 1])])
            transition_score = torch.stack([self.transition[tags[t + 1, b], tags[t, b]] for b in range(batch_size)])
            scores += (emit_score + transition_score) * mask[t]
            transition_to_end = torch.stack([self.transition[self.tag2idx[self.END_TAG], tag[mask[:, b].sum().long()]] for b, tag in enumerate(tags.transpose(0, 1))])
            scores += transition_to_end
        return scores

    def viterbi_decode(self, feats, mask):
        """
        :param feats: (seq_len, batch_size, tag_size)
        :param mask: (seq_len, batch_size)
        :return best_path: (seq_len, batch_size)
        """
        seq_len, batch_size, tag_size = feats.size()
        # initialize scores in log space
        scores = feats.new_full((batch_size, tag_size), fill_value=-10000)
        scores[:, self.tag2idx[self.START_TAG]] = 0
        pointers = []
        # forward
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            scores_t = scores.unsqueeze(1) + self.transition.unsqueeze(0)  # (batch_size, tag_size, tag_size)
            # max along current_tag to obtain: next_tag score, current_tag pointer
            scores_t, pointer = torch.max(scores_t, -1)  # (batch_size, tag_size), (batch_size, tag_size)
            scores_t += feat
            pointers.append(pointer)
            mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
            scores = scores_t * mask_t + scores * (1 - mask_t)
        pointers = torch.stack(pointers, 0) # (seq_len, batch_size, tag_size)
        scores += self.transition[self.tag2idx[self.END_TAG]].unsqueeze(0)
        best_score, best_tag = torch.max(scores, -1)  # (batch_size, ), (batch_size, )
        # backtracking
        best_path = best_tag.unsqueeze(-1).tolist() # list shape (batch_size, 1)
        for i in range(batch_size):
            best_tag_i = best_tag[i]
            seq_len_i = int(mask[:, i].sum())
            for ptr_t in reversed(pointers[:seq_len_i, i]):
                # ptr_t shape (tag_size, )
                best_tag_i = ptr_t[best_tag_i].item()
                best_path[i].append(best_tag_i)
            # pop first tag
            best_path[i].pop()
            # reverse order
            best_path[i].reverse()
        return best_path


class LSTMCRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_size, hidden_size, 
               dropout, token2idx, PAD, tag2idx, START_TAG, END_TAG,
               num_layers = 1, with_ln=False, bidirection=True):
        super(LSTMCRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=token2idx[PAD])
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=bidirection)
        self.with_ln = with_ln
        if with_ln:
            self.layer_norm = nn.LayerNorm(hidden_size)
        if bidirection:
            self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)
        else:
            self.hidden2tag = nn.Linear(hidden_size, tag_size)
        self.crf = CRFLayer(tag_size, tag2idx, START_TAG, END_TAG)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.embedding.weight)
        init.xavier_normal_(self.hidden2tag.weight)

    def get_lstm_features(self, seq, mask):
        """
        :param seq: (seq_len, batch_size)
        :param mask: (seq_len, batch_size)
        """
        embed = self.embedding(seq) # (seq_len, batch_size, embedding_size)
        embed = self.dropout(embed)
        embed = nn.utils.rnn.pack_padded_sequence(embed, mask.sum(0).long().cpu(), enforce_sorted=False)
        lstm_output, _ = self.bilstm(embed) # (seq_len, batch_size, hidden_size)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output)
        lstm_output = lstm_output * mask.unsqueeze(-1)
        if self.with_ln:
            lstm_output = self.layer_norm(lstm_output)
        lstm_features = self.hidden2tag(lstm_output) * mask.unsqueeze(-1)  # (seq_len, batch_size, tag_size)
        return lstm_features

    def neg_log_likelihood(self, seq, tags, mask):
        """
        :param seq: (seq_len, batch_size)
        :param tags: (seq_len, batch_size)
        :param mask: (seq_len, batch_size)
        """
        lstm_features = self.get_lstm_features(seq, mask)
        forward_score = self.crf(lstm_features, mask)
        gold_score = self.crf.score_sentence(lstm_features, tags, mask)
        loss = (forward_score - gold_score).sum()
        
        return loss

    def predict(self, seq, mask):
        """
        :param seq: (seq_len, batch_size)
        :param mask: (seq_len, batch_size)
        """
        lstm_features = self.get_lstm_features(seq, mask)
        best_paths = self.crf.viterbi_decode(lstm_features, mask)

        return best_paths


class BertCRF(nn.Module):
    def __init__(self, bert_model, tag_size, tag2idx, START_TAG, END_TAG, 
                 with_lstm = True, lstm_layers=1, bidirection=True,
                 lstm_hid_size=256, dropout=0.2):
        """
        Params:
            bert_model(str),
                the bert model name, like "bert-base-chinese", or the address of stored bert model
            tag_size (int), 
                the # of tags
            tag2idx (dict),
                the mapping of tags and its id
            START_TAG (str), 
                the starting symbol of tags 
            END_TAG (str), 
                the ending symbol of tags
            with_lstm (bool),
                whether to use lstm at the top of Bert, default True
            lstm_layers(int), 
                the # of layers of lstm, default 1, not used while with_lstm = False
            bidirection(bool),
                whether use bidirecitonal lstm or not, default True, not used while with_lstm = False
            lstm_hid_size(int), 
                the hiddensize of lstm nodes, default 256 , not used while with_lstm = False 
            dropout (float), 
                dropout rate
        """
        # super(BertCRF, self).__init__()
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.with_lstm = with_lstm
        if with_lstm:
            self.bilstm = nn.LSTM(input_size=768,
                              hidden_size=lstm_hid_size,
                              num_layers=lstm_layers,
                              dropout=dropout,
                              bidirectional=bidirection,
                              batch_first=True)
            if bidirection:
                self.hidden2tag = nn.Linear(lstm_hid_size * 2, tag_size)
            else:
                self.hidden2tag = nn.Linear(lstm_hid_size, tag_size)
        else:
            self.hidden2tag = nn.Linear(768, tag_size)

        self.crf = CRFLayer(tag_size, tag2idx, START_TAG, END_TAG)
        # freeze the Bert model
        # for name ,param in self.bert.named_parameters():
        #     param.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.hidden2tag.weight)
    
    def get_lstm_features(self, id_tensor, sen_tensor, mask_tensor):
        """
        Parameters
        ----------
        id_tensor : torch.LongTensor
            in shape  b * l 
        sen_tensor : torch.LongTensor
            in shape  b * l .
        mask_tensor : torch.LongTensor
            in shape  b * l 

        Returns
        -------
        lstm_features : torch.Tensor
            in shape  `b * l * h`( h should be the tag size)

        """
        bert_outs, pooled_outs = self.bert.forward(id_tensor, mask_tensor, sen_tensor, 
                                      return_dict=False, output_hidden_states=False) # b * l *h
        if self.with_lstm:
            bert_outs = self.dropout(bert_outs)
            # bert_outs_pad = nn.utils.rnn.pack_padded_sequence(bert_outs.transpose(0,1), 
            #                                                   mask_tensor.sum(1).long().cpu(), 
            #                                                   enforce_sorted=False)
            # lstm_output, _ = self.bilstm(bert_outs_pad) # (seq_len, batch_size, hidden_size)
            # lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output)
            # lstm_output = lstm_output.transpose(0,1) * mask_tensor.unsqueeze(-1)
            lstm_output,_ = self.bilstm(bert_outs)
        else:
            lstm_output = self.dropout(bert_outs)
        lstm_features = self.hidden2tag(lstm_output) * mask_tensor.unsqueeze(-1)
        return lstm_features
    
    def neg_log_likelihood(self, id_tensor, sen_tensor, mask_tensor, tags):
        """
        :param id_tensor: (batch_size, seq_len)
        :param sen_tensor: (batch_size, seq_len)
        :param mask_tensor: (batch_size, seq_len)
        :param tags: l*b
        """
        lstm_features = self.get_lstm_features(id_tensor, sen_tensor, mask_tensor) # b*l*h
        forward_score = self.crf(lstm_features.transpose(0,1), 
                                 mask_tensor.transpose(0,1))
        gold_score = self.crf.score_sentence(lstm_features.transpose(0,1), 
                                             tags.transpose(0,1), 
                                             mask_tensor.transpose(0,1))
        loss = (forward_score - gold_score).sum()
        
        return loss
    
    def predict(self, id_tensor, sen_tensor, mask_tensor):
        """
        :param seq: (batch_size, seq_len)
        :param mask: (batch_size, seq_len)
        """
        lstm_features = self.get_lstm_features(id_tensor, sen_tensor, mask_tensor)
        best_paths = self.crf.viterbi_decode(lstm_features.transpose(0,1), mask_tensor.transpose(0,1))

        return best_paths
            

def compute_forward(model, seq, tags, mask):
    loss = model.neg_log_likelihood(seq, tags, mask)
    batch_size = seq.size(1)
    loss /= batch_size
    loss.backward()
    return loss.item()