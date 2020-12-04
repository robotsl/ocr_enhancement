import torch
import pandas as pd
import numpy as np
import pickle
import torchvision.models as models
# pip install transformers==2.2.2
from transformers import BertTokenizer, AlbertForMaskedLM
import copy
from torch.nn.functional import softmax

pretrained = 'voidful/albert_chinese_tiny'


dim = 50

def get_label_dict():
    f = open('/home/robotsl/workspace/ocr_enhancement/Corrector/chinese_labels', 'rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict

id2char = get_label_dict()
char2id = {x: y for x, y in zip(id2char.values(), id2char.keys())}

def getAllReady():
    id2char = get_label_dict()
    char2id = {x: y for x, y in zip(id2char.values(), id2char.keys())}
    M = torch.load(f'/home/robotsl/workspace/ocr_enhancement/Corrector/shape2vec.{dim}d')['shape2vec']
    print(M.shape)
    return M

M = getAllReady()


def getTopKSimDict(K=20):
    sim_dct = {}
    for q_char in char2id.keys():
        id = char2id[q_char]
        sim_lst = []
        for i in range(M.shape[0]):
            sim = torch.cosine_similarity(M[id], M[i], dim=-1)
            sim_lst.append((sim, i))
        sim_lst.sort(key=lambda x: x[0], reverse=True)
        topK_id = sim_lst[1:K+1]
        topK_char = []
        for _, id in topK_id:
            topK_char.append(id2char[id])
        sim_dct[q_char] = topK_char
    return sim_dct

sim_dct = getTopKSimDict()
torch.save(sim_dct, "/home/robotsl/workspace/ocr_enhancement/Corrector//sim_dct")



# 加载
def get_sim_dict():
    sim_dict = torch.load('/home/robotsl/workspace/ocr_enhancement/Corrector/sim_dct')
    return sim_dict

sim_dct = get_sim_dict()

# 字间相似度
def charSim(c1="于", c2="干"):
    # if not exists
    if char2id.get(c1) is None or char2id.get(c2) is None:
        return
    sim = torch.cosine_similarity(M[char2id[c1]], M[char2id[c2]], dim=-1)
    return sim



def initial():


def correctAll(sent=""):
    assert(len(sent) > 1)
    for i in range(len(sent)):
        msk_char = sent[i]
        msk_sent = sent[:i] + "[MASK]" + sent[i+1:]
        if msk_sent is not None:
            maskpos = tokenizer.encode(msk_sent, add_special_tokens=True).index(103)

            input_ids = torch.tensor(tokenizer.encode(msk_sent, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss, prediction_scores = outputs[:2]

            logit_prob = softmax(prediction_scores[0, maskpos], dim=0).data.tolist()

            _, indices = torch.topk(prediction_scores[0, maskpos], k=10, dim=0)

            for idx in indices:
                idx = idx.item()
                predicted_token = tokenizer.convert_ids_to_tokens([idx])[0]
                sim = charSim(c1=msk_char, c2=predicted_token)
                if sim is not None and sim > 0.5:
                    if sent[i] != predicted_token:
                        print(f"{sent[i]} -> {predicted_token}")
                    sent = sent[:i] + predicted_token + sent[i+1:]
                    break
    return sent
