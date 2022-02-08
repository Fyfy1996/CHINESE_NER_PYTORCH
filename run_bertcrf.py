# -*- coding: utf-8 -*-


import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

from model import BertCRF
from dataset import crfDataset, prepare_xbatch_for_bert, _prepare_data
from transformers import BertTokenizer, AdamW
import datetime
import os
from lstmcrf_utils import bert_evaluate, save_parser
import argparse


def parse():
    parser = argparse.ArgumentParser("This is the model for BERT+CRF")
    parser.add_argument('--model_name', type=str, default="bertcrf", help="Model name, will create a fold to store model file")
    parser.add_argument('--bert_model_path', type=str, default=os.path.join("pretrained_models","bert-base-chinese"),
                        help="Bert pretrained model files")
    parser.add_argument('--bert_tokenizer_path', type=str, default=os.path.join("pretrained_models","bert-base-chinese","vocab"),
                        help="Bert pretrained tokenizer files")
    parser.add_argument('--train_data_path', type=str, default="dataset/train_data",
                        help="train data path")
    parser.add_argument('--test_data_path', type=str, default="dataset/test_data",
                       help="test data path")
    parser.add_argument('--is_cuda', type=bool, default=True, help="Using cuda or not")
    parser.add_argument('--cuda_device', type=int, default=0, help="When using gpu, use the ith one")
    parser.add_argument('--seed', type=int, default=2021, help="Random seed")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")

    parser.add_argument('--with_lstm', type=bool, default=False, help="Using lstm on top of bert or not")
    parser.add_argument('--rnn_layer', type=int, default=1, help="The number of lstm layers on top of bert, only useful when with_lstm = True")
    parser.add_argument('--lstm_hid_size', type=int, default=256,
                        help="The size of lstm hidden states on top of bert, only useful when with_lstm = True")
    parser.add_argument('--lstm_bidirectional', type=bool, default=True,
                        help="Bidirectional lstm or not on top of bert, only useful when with_lstm = True")
    parser.add_argument('--max_len', type=int, default=256, help="seq len")
    
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate anywhere")
    parser.add_argument('--with_layer_norm', type=bool, default=True, help="layer normalization")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate")
    parser.add_argument('--crf_lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--epochs', type=int, default=20, help="Training epochs")
    parser.add_argument('--log_interval', type=int, default=10, help="Printing things every x steps")
    parser.add_argument('--save_interval', type=int, default=30, help="Saving models every x steps")
    parser.add_argument('--valid_interval', type=int, default=60, help="validation every x steps")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--load_chkpoint', type=bool, default=False, help="load check points or not for further training")
    parser.add_argument('--chkpoint_model', type=str, default="bertcrf/newest_model", help="The newest model which will be continued to be trained")
    parser.add_argument('--chkpoint_optim', type=str, default="bertcrf/newest_optimizer",
                        help="The newest model's optimizer which will be continued to be trained")
    args = parser.parse_args()
    return args

def main(args):
    
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
    # tb_writer = SummaryWriter(args.model_name)

    id2tag = {v: k for k, v in tag2idx.items()}
    if not os.path.exists(args.model_name): 
        os.makedirs(args.model_name)
    save_parser(args, os.path.join(args.model_name, "parser_config.json"))
    

    # set cuda device and seed
    use_cuda = torch.cuda.is_available() and args.is_cuda
    cuda_device = ":{}".format(args.cuda_device)
    device = torch.device('cuda' + cuda_device if use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    print("Loading Datasets")
    train_set = crfDataset(args.train_data_path)  # os.path.join(args.data_path, "train_data"))
    test_set = crfDataset(args.test_data_path)  # os.path.join(args.data_path, "test_data"))
    train_loader = DataLoader(train_set, args.batch_size,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, args.batch_size,
                             num_workers=0, pin_memory=True)

    print("Building models")
    print("model_add: {}".format(args.bert_model_path))
    model = BertCRF(args.bert_model_path, len(tag2idx), tag2idx, START_TAG, END_TAG,
                    with_lstm=args.with_lstm, lstm_layers=args.rnn_layer, bidirection=args.lstm_bidirectional,
                    lstm_hid_size=args.lstm_hid_size, dropout=args.dropout)
    if args.load_chkpoint:
        print("==Loading Model from checkpoint: {}".format(args.chkpoint_model))
        model.load_state_dict(torch.load(args.chkpoint_model))
    print(model)
    model.to(device)

    crf_params = list(map(id, model.crf.parameters()))
    base_params = filter(lambda p: id(p) not in crf_params, model.parameters())
    
    optimizer = AdamW([{"params":base_params},
                       {"params":model.crf.parameters(),"lr":args.crf_lr}],
                       lr=args.lr)
    if args.load_chkpoint:
        print("==Loading optimizer from checkpoint: {}".format(args.chkpoint_optim))
        optimizer.load_state_dict(torch.load(args.chkpoint_optim))

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)

    print("Training", datetime.datetime.now())
    print("Cuda Usage: {}, device: {}".format(use_cuda, device))

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
        print("Start epoch {}".format(eidx).center(60, "="))

        for bidx, batch in enumerate(train_loader):
            model.train()
            x_batch, y_batch = batch[0], batch[1]
            input_ids, segment_ids, mask = prepare_xbatch_for_bert(x_batch, tokenizer, max_len=args.max_len, 
                                              batch_first=True, device=device)
 
            y_batch = _prepare_data(y_batch, tag2idx, END_TAG, device, max_len=args.max_len, batch_first=True)


            optimizer.zero_grad()
            loss = model.neg_log_likelihood(input_ids, segment_ids, mask, y_batch)
            batch_size = input_ids.size(1)
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
                f1, precision, recall = bert_evaluate(model, test_loader, tokenizer, 
                                                      START_TAG, END_TAG, id2tag,
                                                      device=device, mtype="crf")
                # tb_writer.add_scalar("eval/f1", f1, step)
                # tb_writer.add_scalar("eval/precision", precision, step)
                # tb_writer.add_scalar("eval/recall", recall, step)
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


if __name__ == "__main__":

    args = parse()
    save_parser(args, os.path.join(args.model_name, "parser_config.json"))
    main(args)
