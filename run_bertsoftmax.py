import os
import datetime
import argparse
import torch
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from torch.utils.data import DataLoader
from transformers import AdamW

from dataset import crfDataset, prepare_xbatch_for_bert, _prepare_data
from lstmcrf_utils import bert_evaluate, save_parser



def parser():
    parser = argparse.ArgumentParser("This is a trying on argparse")
    parser.add_argument('--task_name', type=str, 
                        help="Model name, will create a fold to store model file")
    parser.add_argument('--bert_model_path', type=str, default=os.path.join("pretrained_models","bert-base-chinese"),
                        help="Bert pretrained model files")
    parser.add_argument('--bert_tokenizer_path', type=str, default=os.path.join("pretrained_models","bert-base-chinese","vocab"),
                        help="Bert pretrained tokenizer files")
    
    parser.add_argument('--train_data_path', type=str, default="dataset/train_data",
                        help="train data path")
    parser.add_argument('--test_data_path', type=str, default="dataset/test_data",
                       help="test data path")
    
    parser.add_argument('--max_len', type=int, default=256, help="seq len")
    parser.add_argument('--use_cuda', type=bool, default=True, help="Using cuda or not")
    parser.add_argument('--cuda_device', type=int, default=0, help="When using gpu, use the ith one")
    parser.add_argument('--seed', type=int, default=2021, help="Random seed")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")

    parser.add_argument('--lr', type=float, default=3e-5, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0, help="learning rate")
    
    parser.add_argument('--epochs', type=int, default=20, help="Training epochs")
    parser.add_argument('--log_interval', type=int, default=10, help="Printing things every x steps")
    parser.add_argument('--save_interval', type=int, default=30, help="Saving models every x steps")
    parser.add_argument('--valid_interval', type=int, default=60, help="validation every x steps")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--load_chkpoint', type=bool, default=False, help="load check points or not for further training")
    parser.add_argument('--chkpoint_model', type=str, help="The newest model which will be continued to be trained")
    parser.add_argument('--chkpoint_optim', type=str,
                        help="The newest model's optimizer which will be continued to be trained")
    args = parser.parse_args()
    return args

# class arguments:
#     def __init__(self):
#         self.task_name = "bertsoftmax_ner"
#         self.model_path = "pretrained_models/bert-base-chinese"
#         self.bert_tokenizer_path = os.path.join("pretrained_models","bert-base-chinese", "vocab")

#         self.train_data_path = "dataset/train_data"
#         self.test_data_path = "dataset/test_data"
#         self.max_len = 512

#         self.use_cuda = True
#         self.cuda_device = 0
#         self.seed = 1234
#         self.batch_size = 256

#         self.lr = 2.5e-5
#         self.weight_decay = 0

#         self.epochs = 30
#         self.log_interval = 30
#         self.save_interval = 300
#         self.valid_interval = 300
#         self.patience = 30
#         self.load_chkpoint = False
#         self.chkpoint_model = os.path.join(self.task_name, "best_model")
#         self.chkpoint_optim = os.path.join(self.task_name, "best_optimizer")



def main(args):
    if not os.path.exists(args.task_name):
        os.mkdir(args.task_name)
    START_TAG, END_TAG, PadTag = "<START_TAG>", "<END_TAG>", "O"
    O = "O"
    BLOC = "B-LOC"
    ILOC = "I-LOC"
    BORG = "B-ORG"
    IORG = "I-ORG"
    BPER = "B-PER"
    IPER = "I-PER"
    tag2idx = {
        START_TAG: 0,
        END_TAG  : 1,
        O: 2,
        BLOC: 3,
        ILOC: 4,
        BORG: 5,
        IORG: 6,
        BPER: 7,
        IPER: 8
    }
    id2tag = {v:k for k,v in tag2idx.items()}


    # prepare Dataloader
    train_set = crfDataset(args.train_data_path)
    test_set = crfDataset(args.test_data_path)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)


    # Prepare device
    is_cuda = torch.cuda.is_available() & args.use_cuda
    device = torch.device("cuda:{}".format(args.cuda_device) if is_cuda else "cpu")

    # set torch seed
    torch.manual_seed(args.seed)
    if is_cuda:
        torch.cuda.manual_seed(args.seed)

    # Prepare tokenizer, model and optimizers
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    config = BertConfig.from_pretrained(os.path.join(args.model_path,"config.json"))
    config.num_labels = len(tag2idx)
    model = BertForTokenClassification.from_pretrained(os.path.join(args.model_path,"pytorch_model.bin"), config=config)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)



    model.train()
    step = -1
    patience = 0 # for early stopping
    best_f1 = 0
    early_stop = False
    print("Training", datetime.datetime.now())
    print("Cuda Usage: {}, device: {}".format(is_cuda, device))
    for epoch in range(1, args.epochs+1):
        print("Start epoch {}".format(epoch).center(60, "="))
        if early_stop:
            print("Early stop. epoch {} step {} best f1 {}".format(epoch, step, best_f1))
            break
        for bidx, (text, labels) in enumerate(train_loader):
            step += 1
            optimizer.zero_grad()
            
            # text = [ "[CLS]"+" "+k+" "+"[SEP]" for k in text]
            # labels = [ START_TAG+" "+label+" "+END_TAG for label in labels ]
            x_batch = prepare_xbatch_for_bert(text, tokenizer, max_len=args.max_len, 
                                              batch_first=True, device=device)
            # y_batch = prepare_labels(labels, tag2idx, StartTag, EndTag, PadTag, max_len=args.max_len, return_tensors="pt", device=device)
            y_batch = _prepare_data(labels, tag2idx, "O", "O", device, max_len=args.max_len, batch_first=True)
            outputs = model(input_ids=x_batch[0], token_type_ids=x_batch[1],
                            attention_mask=x_batch[2], labels = y_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if step % args.log_interval == 0:
                print("epoch {} step {} batch {} loss {}".format(epoch, step, bidx, loss.item()))
            if step % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.task_name, "newest_model"))
                torch.save(optimizer.state_dict(), os.path.join(args.task_name, "newest_optimizer"))
            if step % args.valid_interval == 0:
                f1, precision, recall = bert_evaluate(model, test_loader, tokenizer, START_TAG, END_TAG, id2tag, device=device, mtype="softmax")
                print("[valid] epoch {} step {} f1 {} precision {} recall {}".format(epoch, step, f1, precision, recall))
                if f1 > best_f1:
                    patience = 0
                    best_f1 = f1
                    torch.save(model.state_dict(), os.path.join(args.task_name, "best_model"))
                    torch.save(optimizer.state_dict(), os.path.join(args.task_name, "best_optimizer"))
                else:
                    patience += 1
                    if patience == args.patience:
                        early_stop = True

if __name__ == "__main__":
    args = parser()
    save_parser(args, os.path.join(args.task_name, "parser_config.json"))
    main(args)
    

