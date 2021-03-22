# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:35:31 2021

@author: fanyong
"""
import logging
import sys
import os
import tqdm

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch import nn, optim

from dataset import read_corpus, read_dictionary, vocab_build, crfDataset, prepare_databatches
from model import LSTMCRF, compute_forward
from lstmcrf_utils import evaluate, get_entity

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
        self.save_interval = 30
        self.valid_interval = 60
        self.patience = 30
        self.load_chkpoint = True
        self.chkpoint_model = "lstmcrf/newest_model"
        self.chkpoint_optim = "lstmcrf/newest_optimizer"
        
        
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
    
    # build vocab, word2id, id2tag, id2word
    vocab_build(args.vocab_path, args.train_data_path, 10, token2idx)
    word2id = read_dictionary(args.vocab_path)
    id2word = {v:k for k,v in word2id.items()}
    id2tag  = {v:k for k,v in tag2idx.items()} 
    
    # set cuda device and seed
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        
    # load datasets
    print("Loading Datasets")
    train_set = crfDataset(args.train_data_path)#os.path.join(args.data_path, "train_data"))
    test_set  = crfDataset(args.test_data_path)#os.path.join(args.data_path, "test_data"))
    train_loader = DataLoader(train_set, args.batch_size, shuffle=False, 
                              num_workers=1, pin_memory=True )
    test_loader  = DataLoader(test_set, args.batch_size, shuffle=False, 
                              num_workers=1, pin_memory=True )
    
    
    # Building Model
    print("Building model")
    
    model = LSTMCRF(vocab_size=len(word2id), tag_size=len(tag2idx), embedding_size=args.embedding_size,
                    hidden_size=args.hidden_size, dropout = args.dropout, 
                    token2idx=word2id, PAD=PAD, tag2idx=tag2idx, START_TAG=START_TAG, END_TAG=END_TAG,
                    num_layers = args.rnn_layer, with_ln=False, bidirection=True)
    if args.load_chkpoint:
        print("==Loading Model from checkpoint: {}".format(args.chkpoint_model))
        model.load_state_dict(torch.load(args.chkpoint_model))
    
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.load_chkpoint:
        print("==Loading Model from checkpoint: {}".format(args.chkpoint_optim))
        optimizer.load_state_dict(torch.load(args.chkpoint_optim))
    
    
    print("Start training")
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
        # with tqdm.tqdm(total = len(train_loader)) as pbar:
        for bidx, batch in enumerate(train_loader):
#            seq = _prepare_data(batch[0], token2idx, PAD, device)
#            tags = _prepare_data(batch[1], tag2idx, END_TAG, device)
#            mask = torch.ne(seq, float(token2idx[PAD])).float()
#            length = mask.sum(0)
#            _, idx = length.sort(0, descending=True)
#            seq = seq[:, idx]
#            tags = tags[:, idx]
#            mask = mask[:, idx]
            seq, tags, mask = prepare_databatches(batch[0], batch[1], word2id, PAD, tag2idx,
                                                END_TAG, UNK, device=device)
            optimizer.zero_grad()
            loss = compute_forward(model, seq, tags, mask)
            tb_writer.add_scalar("train/loss", loss, step)
            tb_writer.add_scalar("train/epoch", step, eidx)
            optimizer.step()
            # pbar.update(1)
    
            step += 1
            if step % args.log_interval == 0:
                print("epoch {} step {} batch {} loss {}".format(eidx, step, bidx, loss))
            if step % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.model_name, "newest_model"))
                torch.save(optimizer.state_dict(), os.path.join(args.model_name, "newest_optimizer"))
            if step % args.valid_interval == 0:
                f1, precision, recall = evaluate(model, test_loader,word2id, PAD, id2tag, UNK,device)
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
                        
                        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

