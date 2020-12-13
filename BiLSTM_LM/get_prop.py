from model import BiLSTM # don't delete it!!
from val_utils import *
import torch
device = torch.device('cuda')

input_sentence = '数[mask]'# input your sentence here and using "[mask]"去get the target word proability
model = torch.load("./model/LM_128_withsoftmaxbn.pt").to(device)

def de_mask(input_sentence):
    input = list(input_sentence)
    for i in range(len(input)):
        if i + 2 <= len(input) and input[i] == '[' and input[i + 1] == 'm' and input[i + 2] == 'a':
            input[i] = '/'# '/'will be processed as <unk>
            flag = i-1
            del input[i+1:i + 6]
    return input, flag

def get_prob(input_sentence, flag):

    L = len(list(input_sentence))
    input = torch.zeros([1, L], dtype=torch.int64).to(device)

    input_indices, _ = sentences_to_indices(input_sentence, w2i, L)

    for i in range(L):
        input[0][i] = input_indices[i]

    out = model(input)

    indices = list(torch.argsort(out[flag], descending=True)[0:9].cpu().detach().numpy())

    for idx in range(len(indices)):
        key = indices[idx]
        word = i2w[key]
        prob = out[flag][key].cpu().detach().numpy()
        print(prob, word)

s, flag = de_mask(input_sentence)
get_prob(s, flag)

