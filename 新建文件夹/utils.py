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
i2w['0'] = '<unk>'
w2v_m['<unk>'] = np.zeros([300])

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def read_txt(data_path):
    with open(data_path, 'rb') as f:
        x, x_cache = [], []
        l = []
        contents = f.read().decode()
        for raw in re.split('\s{2,}', contents):
        # X就选择为raw,而Y比X快一个值，最后一个为‘\n’
            x_cache.append(raw)
            x_str_cache = list(raw)
            if x_str_cache[0] != '<':
                x.append(raw)
                x_str = list(raw)
                l.append(len(x_str))

        l = np.asarray(l)
        x = np.asarray(x)
        return l, x

def sentences_to_indices(sentence, w2i, Len):

    # 使用0初始化X_indices
    X_indices = np.zeros((Len), dtype=np.float32)
    Y_indices = np.zeros_like(X_indices)
    sentences_words = list(sentence)
    count = 0
    # 初始化j为0
    # 遍历这个单词列表
    for j, w in enumerate(sentences_words):
        # 将X_indices的第j号元素为对应的单词索引
        if w in w2i.keys():
            X_indices[j] = w2i[w]
        else:
            X_indices[j] = w2i['<unk>']
            count += 1
        if j > 0:
            Y_indices[j-1] = X_indices[j]
    if j-2 > 0:
        Y_indices[j-1] = w2i['\\']
    #print('The number of UNK is :', count)
    X, Y = torch.LongTensor(X_indices), torch.LongTensor(Y_indices)
    return X, Y