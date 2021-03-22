# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:48:13 2021

@author: fanyong
"""


from model import LSTMCRF
from dataset import read_dictionary
import torch
from torch.utils.data import DataLoader
from dataset import crfDataset
import os
from dataset import _prepare_data
from lstmcrf_utils import get_entity

class argss:
    def __init__(self):
        self.model_name = "lstmcrf"
        self.train_data_path = "dataset/train_data"
        self.test_data_path = "dataset/test_data"
        self.vocab_path = "vocab.pkl"
        self.no_cuda = False
        self.seed = 2021
        self.batch_size = 64
        self.embedding_size = 128
        self.hidden_size = 128
        self.rnn_layer = 1
        self.dropout = 0.2
        self.with_layer_norm = False
        self.lr = 0.0005
        self.epochs = 50
        self.log_interval = 10
        self.save_interval = 10
        self.valid_interval = 30
        self.patience = 30

args = argss()
START_TAG = "<START_TAG>"
END_TAG = "<END_TAG>"
O = "O"
BLOC = "B-LOC"
ILOC = "I-LOC"
BORG = "B-ORG"
IORG = "I-ORG"
BPER = "B-PER"
IPER = "I-PER"
PAD = "<PAD>"
UNK = "<UNK>"
token2idx = {
  PAD: 0,
  UNK: 1
}
tag2idx = {
  START_TAG: 0,
  END_TAG: 1,
  O: 2,
  BLOC: 3,
  ILOC: 4,
  BORG: 5,
  IORG: 6,
  BPER: 7,
  IPER: 8
}
id2tag = {v:k for k,v in tag2idx.items()}

word2id = read_dictionary(args.vocab_path)
model_path = os.path.join("lstmcrf","best_model")
tag_size = len(tag2idx)
model = LSTMCRF(vocab_size=len(word2id), tag_size=len(tag2idx), embedding_size=args.embedding_size,
                hidden_size=args.hidden_size, dropout = args.dropout, 
                token2idx=word2id, PAD=PAD, tag2idx=tag2idx, START_TAG=START_TAG, END_TAG=END_TAG,
                num_layers = args.rnn_layer, with_ln=False, bidirection=True)


model.load_state_dict(torch.load("D:/nlp/SELF/Edited_bilstm_crf_pytorch/lstmcrf/best_model"))


while True:
    sample_sent = input("Plz enter a sentence:")
    if sample_sent == "Exit":
        break
    sent = (" "+" ".join([w for w in sample_sent.strip()]) ,)
    print(sent)
    model.eval()
    with torch.no_grad():
        seq  = _prepare_data(sent, word2id, PAD, UNK)
        mask = torch.ne(seq, float(token2idx[PAD])).float()
        length = mask.sum(0)
        _, idx = length.sort(0, descending=True)
        seq = seq[:, idx]
        mask = mask[:, idx]
        best_path = model.predict(seq, mask)
        # hyp =   [ id2tag[i[0]] for i in best_path]
        hyp = [id2tag[i] for i in best_path[0]]
        entity_idx = get_entity(hyp)
        #print(best_path)
        #print(hyp)
        #print(entity_idx)
        for beg, end in entity_idx:
            print(sample_sent[beg:end+1])
    

