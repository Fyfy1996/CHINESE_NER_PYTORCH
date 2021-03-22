# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:06:27 2021

@author: fanyong
"""


import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import BertCRF
from dataset import crfDataset, prepare_xbatch_for_bert, _prepare_data
from pytorch_pretrained_bert import BertTokenizer
import datetime
import os
from lstmcrf_utils import bert_evaluate

class argss:
    def __init__(self):
        self.model_name = "bertcrf"
        self.bert_model_path = os.path.join("pretrained_models","bert-base-chinese")
        self.bert_tokenizer_path = "bert-base-chinese" # os.path.join("pretrained_models","bert-base-chinese", "vocab")
        
        self.train_data_path = "dataset/train_data"
        self.test_data_path = "dataset/test_data"
        self.vocab_path = "vocab.pkl"
        self.is_cuda = True
        self.seed = 2021
        self.batch_size = 2
        self.embedding_size = 128
        self.hidden_size = 128
        self.rnn_layer = 1
        self.dropout = 0.2
        self.with_layer_norm = False
        self.lr = 0.0005
        self.epochs = 50
        self.log_interval = 10
        self.save_interval = 30
        self.valid_interval = 60
        self.patience = 30
        self.load_chkpoint = False
        self.chkpoint_model = os.path.join(self.model_name,"newest_model")
        self.chkpoint_optim = os.path.join(self.model_name,"newest_optimizer")



if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
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
    args = argss()
    tb_writer = SummaryWriter(args.model_name)

    id2tag ={v:k for k,v in tag2idx.items()}
    
    if not os.path.exists(args.model_name):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(args.model_name)
    
    # set cuda device and seed
    use_cuda = torch.cuda.is_available() and not args.is_cuda
    device = torch.device('cuda' if use_cuda else 'cpu') 
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        
    print("Loading Datasets")
    train_set = crfDataset(args.train_data_path)#os.path.join(args.data_path, "train_data"))
    test_set  = crfDataset(args.test_data_path)#os.path.join(args.data_path, "test_data"))
    train_loader = DataLoader(train_set, args.batch_size, shuffle=False, 
                              num_workers=0, pin_memory=True )
    test_loader  = DataLoader(test_set, args.batch_size, shuffle=False, 
                              num_workers=0, pin_memory=True )
    
    print("Building models")
    
    model = BertCRF(args.bert_model_path, len(tag2idx), tag2idx, START_TAG, END_TAG, 
                     with_lstm = False, lstm_layers=1, bidirection=True,
                     lstm_hid_size=256, dropout=0.2)
    if args.load_chkpoint:
        print("==Loading Model from checkpoint: {}".format(args.chkpoint_model))
        model.load_state_dict(torch.load(args.chkpoint_model))
    print(model)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.load_chkpoint:
        print("==Loading optimizer from checkpoint: {}".format(args.chkpoint_optim))
        optimizer.load_state_dict(torch.load(args.chkpoint_optim))
    
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    
    
    print("Training", datetime.datetime.now())
    print("Cuda Usage: {}, device: {}".format(use_cuda, device))
    model.train()
    step = 0
    
    best_f1 = 0
    patience = 0
    early_stop = False

    for eidx in range(1, args.epochs + 1):
        if eidx == 2:
            model.debug = True
        if early_stop:
            print("Early stop. epoch {} step {} best f1 {}".format(eidx, step, best_f1))
            break
            # sys.exit(0)
        print("Start epoch {}".format(eidx).center(60,"="))
    
        for bidx, batch  in enumerate(train_loader):
            x_batch, y_batch = batch[0], batch[1]
            input_ids, segment_ids, mask = prepare_xbatch_for_bert(x_batch, tokenizer)
            # input_ids = input_ids.to(device)
            # segment_ids = input_ids.to(device)
            # mask = mask.to(device)
            
            y_batch = [START_TAG+ " " + line + " " +END_TAG for line in y_batch]
            tags = _prepare_data(y_batch, tag2idx, END_TAG, device)
            # tags = tags.to(device)
            
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(input_ids, segment_ids, mask, tags)
            batch_size = input_ids.size(0)
            loss /= batch_size
            # print(loss)
            loss.backward() 
            optimizer.step()
            # break
            step += 1
            if step % args.log_interval == 0:
                print("epoch {} step {} batch {} loss {}".format(eidx, step, bidx, loss))
            if step % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.model_name, "newest_model"))
                torch.save(optimizer.state_dict(), os.path.join(args.model_name, "newest_optimizer"))
            if step % args.valid_interval == 0:
                f1, precision, recall = bert_evaluate(model, test_loader, tokenizer, START_TAG, END_TAG, id2tag, device=None)
                tb_writer.add_scalar("eval/f1", f1, step)
                tb_writer.add_scalar("eval/precision", precision, step)
                tb_writer.add_scalar("eval/recall", recall, step)
                print("[valid] epoch {} step {} f1 {} precision {} recall {}".format(eidx, step, f1, precision, recall))
                if f1 > best_f1:
                    patience = 0
                    best_f1 = f1
                    torch.save(model.state_dict(), os.path.join(args.model_name, "best_model"))
                    torch.save(optimizer.state_dict(), os.path.join(args.model_name, "best_optimizer"))
                else:
                    patience += 1
                    if patience == args.patience:
                        early_stop = True
    
    
    
    
    
    
    
    
    
    