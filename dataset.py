# -*- coding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data


def vocab_build(vocab_path, corpus_path, min_count, symbol2idx = { "<PAD>": 0,
                                                                   "<UNK>": 1} ):
    """
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)

    word2id = {}
    max_len = 0
    for sent_, tag_ in data:
        if max_len < len(sent_):
            max_len = len(sent_)
        for word in sent_:
            word2id[word] = word2id.get(word,0) + 1
    # return word2id
    low_freq_words = []
    for word, word_freq in word2id.items():
        if word_freq < min_count:
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = len(symbol2idx)
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    # word2id['<UNK>'] = new_id
    # word2id['<PAD>'] = 0
    
    print("The max length of sentence is {}".format(max_len))
    print("The length of vocabs is {}".format(len(word2id)))
    with open(vocab_path, 'wb') as fw:
        pickle.dump({**symbol2idx, **word2id}, fw)
    print("Successfully build the vocab dict in {}".format(vocab_path))
#    
#    return max_len

def read_dictionary(vocab_path):
    """
    :param vocab_path:
    :return:
    """
    # vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id



class crfDataset(Dataset):
    def __init__(self, data_path): # , word2id, tag2id, word_unk="<UNK>"):
        self.data = read_corpus(data_path)
#        self.word2id = word2id
#        self.tag2id = tag2id
#        self.word_unk = word_unk
        
    def __getitem__(self, index):
        sent, tags = self.data[index]
        sent = " ".join(sent)
        tags = " ".join(tags)
#        sent_id = [ self.word2id[word] if word in self.word2id else self.word2id[self.word_unk] for word in sent]
#        tag_id  = [ self.tag2id[tag]   for tag in tags]
        return sent, tags
        
    def __len__(self):
        return len(self.data)
    
def _prepare_data(samples, vocab, pad, UNK, device=None, max_len=None, batch_first=False):
#     
    """
    Transfer str/tag to ids for words/tags
    Returning seq in seq_len * batch_len
    """
    samples = list(map(lambda s: s.strip().split(" "), samples))
    batch_size = len(samples)
    sizes = [len(s) for s in samples]
    if max_len == None:
        max_len = max(sizes)
    x_np = np.full((batch_size, max_len), fill_value=vocab[pad], dtype='int64')
    for i in range(batch_size):
        x_np[i, :sizes[i]] = [vocab[token] if token in vocab else vocab[UNK] for token in samples[i]]
    if batch_first:
        return torch.LongTensor(x_np).to(device)
    else:
        return torch.LongTensor(x_np.T).to(device)


def prepare_databatches(batch_x, batch_y, token2idx, PAD, tag2idx, END_TAG, UNK, device=None):
    """
    Prepare the tensors for a batch
    Returning seq in seq_len * batch_len
    """
    seq = _prepare_data(batch_x, token2idx, PAD, UNK, device)
    tags = _prepare_data(batch_y, tag2idx, END_TAG, UNK, device)
    mask = torch.ne(seq, float(token2idx[PAD])).float()
    length = mask.sum(0)
    _, idx = length.sort(0, descending=True)
    seq = seq[:, idx]
    tags = tags[:, idx]
    mask = mask[:, idx]
    return seq, tags, mask
    
    
def prepare_xbatch_for_bert(x_batch, tokenizer, batch_first=False, device=None, max_len=256):
    """
    Params:
        x_batch (tuple), the x part of one iter of a DataLoader, eg.("我 叫 汤 姆", "我 喜 欢 笑") 
        tokenizer, the class of tokenizer from pytorch_pretrained_bert
        batch_first (bool), if True, return the 3 tensors in  size batch_size * seq_size * dim,
                            else seq_size * batch_size * dim
    Returns:
        id_tensor, the id tensor for bert model
        sen_tensor, the segment id tensor for bert model
        mask_tensor, the mask tensor for bert model inputs
    
    """
    # batch_size = len(x_batch)
    # str_batch =  [ ["[CLS]"] + tokenizer.tokenize(k.replace(" ","")) + ["[SEP]"] for k in x_batch]
    str_batch = [tokenizer.tokenize(k.replace(" ","")) for k in x_batch]
    # max_len = max([len(line) for line in str_batch])
    
    input_ids = []
    segment_ids = []
    input_mask = []
    for line in str_batch:
        line_ids = tokenizer.convert_tokens_to_ids(line)
        line_segmnt = [0] * len(line_ids)
        line_mask = [1] * len(line_ids)
        paddings = [0] * max(max_len - len(line_ids), 0)
        input_ids.append(line_ids + paddings)
        segment_ids.append(line_segmnt + paddings)
        input_mask.append(line_mask + paddings)
    id_tensor =  torch.LongTensor(input_ids)
    sen_tensor=  torch.LongTensor(segment_ids)
    mask_tensor= torch.LongTensor(input_mask)
    if not batch_first:
        id_tensor = id_tensor.transpose(0,1)
        sen_tensor = sen_tensor.transpose(0,1)
        mask_tensor = mask_tensor.transpose(0,1)
    # assert (id_tensor.size()[0] == batch _size) & (id_tensor.size()[1] ==  max_len), "Check for input dim"
    return (id_tensor.to(device), sen_tensor.to(device), mask_tensor.to(device))
    
    
    
    
    
    
    
    
    
    
    
    
