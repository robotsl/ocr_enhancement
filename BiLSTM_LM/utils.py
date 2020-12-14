import torch
import torch.nn as nn
import re
import numpy as np

def pretrained_embedding_layer(w2v_map, w2i):
    vocab_len = len(w2i) + 1
    emb_dim =w2v_map["你"].shape[0]

    # 初始化嵌入矩阵
    emb_matrix = np.zeros((vocab_len, emb_dim))
    # 将嵌入矩阵的每行的“index”设置为词汇“index”的词向量表示
    # 加载一下权重矩阵
    for word, index in w2i.items():
        emb_matrix[index, :] = w2v_map[word]
    # 定义embbeding层
    tensor_emb_matrix = torch.from_numpy(emb_matrix)
    #从预训练好的嵌入矩阵中导入权重
    embedding_layer = nn.Embedding.from_pretrained(tensor_emb_matrix, freeze=True)

    return tensor_emb_matrix

w2i = np.load('./data/w2i_sl.npy', allow_pickle=True).item()
i2w = np.load('./data/i2w_sl.npy', allow_pickle=True).item()
w2v_m = np.load('./data/w2v_m_sl.npy', allow_pickle=True).item()

w2i['<unk>'] = 0
i2w[0] = '<unk>'
w2v_m['<unk>'] = np.random.randn((300))

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def del_unk(sentence, w2i):
    new_sentence = ''
    word_list = list(sentence)
    for word in word_list:
        if word in w2i.keys():
            new_sentence = new_sentence + word
    return new_sentence
#data augmentation
def read_txt(data_path, w2i):
    with open(data_path, 'rb') as f:
        x, x_cache, s_cache = [], [], []
        l = []
        contents = f.read().decode()
        for raw in re.split('\s{2,}', contents):
            #x_cache.append(raw)
            x_str_cache = list(raw)
            raw = del_unk(raw, w2i)
            if len(raw) > 0:
                if x_str_cache[0] != '<':
                    x.append(raw)
                    l.append(len(raw))
                for sen in re.split('[，。]', raw):
                    if len(sen) > 0:
                        x.append(sen)
                        l.append(len(sen))

        l = np.asarray(l)
        x = np.asarray(x)
        return l, x




def sentences_to_indices(sentence, w2i):
    Len = len(list(sentence))
    # 使用0初始化X_indices
    sentences_words = list(sentence)
    X_indices = np.zeros((Len), dtype=np.float32)
    Y_indices = np.zeros_like(X_indices)

    # 初始化j为0
    # 遍历这个单词列表
    for j, w in enumerate(sentences_words):
        # 将X_indices的第j号元素为对应的单词索引
        if w in w2i.keys():
            X_indices[j] = w2i[w]
        #由于词典较小，UNK太多导致结果不好，这里直接过滤掉UNK
       # else:
         #   X_indices[j] = 0 # UNK的索引
         #   count += 1
        if j > 0:
            Y_indices[j-1] = X_indices[j]
    Y_indices[-1] = w2i['\\']#换行
    #print('The number of UNK is :', count)
    X, Y = torch.LongTensor(X_indices), torch.LongTensor(Y_indices)
    return X, Y



