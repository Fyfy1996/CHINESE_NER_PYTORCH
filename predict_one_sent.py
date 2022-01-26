# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:49:54 2022

@author: test1
"""

import torch
# from pytorch_pretrained_bert import BertModel, BertTokenizer

from model import BertCRF
from dataset import prepare_xbatch_for_bert, crfDataset


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
id2tag = {v:i for i,v in tag2idx.items()}

model = BertCRF("pretrained_models/bert-base-chinese", len(tag2idx), tag2idx, START_TAG, END_TAG, 
                 with_lstm=False, lstm_layers=1, bidirection=True,
                 lstm_hid_size=256, dropout=0.1)
#%%
model.load_state_dict(torch.load("models/best_model",map_location=torch.device('cpu')))

#%%
from pytorch_pretrained_bert import BertTokenizer
sent = ["毛 泽 东 在 大 陆 建 立 的 政 权 , 中 共 中 央 。",
        "今 天 ， 在 北 京 ， 我 们 来 自 上 海 的 同 学 欢 聚 一 堂 。 潘 林 轩 如 此 说 道",
        "武 当 山 的 张 天 男 在 新 西 兰 跳 伞 ； 加 入 了 中 国 共 产 党 。"]
tokenizer = BertTokenizer.from_pretrained("pretrained_models/bert-base-chinese")
model.eval()
with torch.no_grad():
    seq, seg, mask = prepare_xbatch_for_bert(sent, tokenizer, batch_first=True)   
    ids = model.predict(seq, seg, mask)

tags = [[id2tag[i] for i in line] for line in ids]
#%%

    