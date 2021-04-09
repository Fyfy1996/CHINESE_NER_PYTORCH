# -*- coding: utf-8 -*-

import torch
from dataset import _prepare_data, prepare_xbatch_for_bert


def get_entity(tags):
    entity = []
    prev_entity = "O"
    start = -1
    end = -1
    for i, tag in enumerate(tags):
        if tag[0] == "O":
            if prev_entity != "O":
                entity.append((start, end))
            prev_entity = "O"
        if tag[0] == "B":
            if prev_entity != "O":
                entity.append((start, end))
            prev_entity = tag[2:]
            start = end = i
        if tag[0] == "I":
            if prev_entity == tag[2:]:
                end = i
    return entity

def evaluate(model, testset_loader, token2idx, PAD, idx2tag, UNK, device):
    model.eval()
    correct_num = 0
    predict_num = 0
    truth_num = 0
    with torch.no_grad():
        for bidx, batch in enumerate(testset_loader):
            seq = _prepare_data(batch[0], token2idx, PAD, UNK, device)
            mask = torch.ne(seq, float(token2idx[PAD])).float()
            length = mask.sum(0)
            _, idx = length.sort(0, descending=True)
            seq = seq[:, idx]
            mask = mask[:, idx]
            best_path = model.predict(seq, mask)
            ground_truth = [batch[1][i].strip().split(" ") for i in idx]
            for hyp, gold in zip(best_path, ground_truth):
                hyp = list(map(lambda x: idx2tag[x], hyp))
                predict_entities = get_entity(hyp)
                gold_entities = get_entity(gold)
                correct_num += len(set(predict_entities) & set(gold_entities))
                predict_num += len(set(predict_entities))
                truth_num += len(set(gold_entities))
    # calculate F1 on entity
    precision = correct_num / predict_num if predict_num else 0
    recall = correct_num / truth_num if truth_num else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    model.train()
    return f1, precision, recall
