from LM import BiLSTM
from utils import sentences_to_indices, w2i, i2w
import torch
import numpy as np
from torch.nn.functional import softmax
device = torch.device('cuda')
#原始的句子
ori_input_sentence = '今天出了太阳'
#识别结果
re_out = '令天出了太阳'
#遮挡的句子
input_sentence = '[mask]天出了太阳'

similar = ['今', '令', '金']# 假设shape2vector结果

model = torch.load("./model/LM.pt").to(device)

def de_mask(input_sentence):
    input = list(input_sentence)
    for i in range(len(input)):
        if i + 2 <= len(input) and input[i] == '[' and input[i + 1] == 'm' and input[i + 2] == 'a':
            input[i] = '/'
            flag = i
            del input[i+1:i + 6]
    return input, flag

def get_prob(input_sentence, flag):
    '''

    :param input_sentence:
    :return:
    '''
    L = len(list(input_sentence))
    input = torch.zeros([1, L], dtype=torch.int64).to(device)

    input_indices, _ = sentences_to_indices(input_sentence, w2i, L)

    for i in range(L):
        input[0][i] = input_indices[i]

    out = softmax(model(input).view(L, 7100), dim=1)

    similar_indices = []
    out_similar = []
    for word in similar:
        similar_indices.append(w2i[word])
        out_similar.append(out[flag][w2i[word]])

    indices = torch.argsort(torch.Tensor(out_similar), descending=True).cpu().detach().numpy()

    for idx in range(len(indices)):
        key = indices[idx]
        word = similar[key]
        prob = out[flag][key].cpu().detach().numpy()
        print(prob, word)

s, flag = de_mask(input_sentence)
get_prob(s, flag)

