from model import BiLSTM
from utils import sentences_to_indices, w2i, i2w
import torch
from torch.nn.functional import softmax
device = torch.device('cuda')

input_sentence = '数[mask]'
model = torch.load("./model/LM_128_withsoftmaxbn.pt").to(device)

def de_mask(input_sentence):
    input = list(input_sentence)
    for i in range(len(input)):
        if i + 2 <= len(input) and input[i] == '[' and input[i + 1] == 'm' and input[i + 2] == 'a':
            input[i] = '/'
            flag = i - 1
            del input[i+1:i + 6]
    return input, flag

def get_prob(input_sentence, flag):
    '''

    :param input_sentence:
    :return:
    '''
    L = len(list(input_sentence))
    input_orig = torch.zeros([1, L], dtype=torch.int64).to(device)
    input = torch.zeros_like(input_orig)
    input_indices_orig, _ = sentences_to_indices(input_sentence, w2i, L)
    for i in range(L):
        input_orig[0][i] = input_indices_orig[i]

    out1 = model(input_orig)
    max_indices = int(torch.argmax(out1[flag]).cpu().detach().numpy())
    input_sentence[flag+1] = i2w[max_indices]
    input_indices, _ = sentences_to_indices(input_sentence, w2i, L)
    # 重复采样二次预测大法，先选择第一轮预测最高的那个字将之前的‘\’替换掉，之后再次输入到网络中进行预测
    # 对一些比较模棱两可的问题可能有用，例如前两个字的首轮概率都接近0.5
    for i in range(L):
        input[0][i] = input_indices[i]

    out = softmax(model(input), dim=1)

    indices = list(torch.argsort(out[flag], descending=True)[0:9].cpu().detach().numpy())

    for idx in range(len(indices)):
        key = indices[idx]
        word = i2w[key]
        prob = out[flag][key].cpu().detach().numpy()
        print(prob, word)

s, flag = de_mask(input_sentence)
get_prob(s, flag)

