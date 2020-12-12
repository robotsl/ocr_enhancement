import torch
import torch.nn as nn
import os
from data import *
import re
from hparams import hparams

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

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def read_txt(data_path):
    with open(data_path, 'rb') as f:
        x = []
        l = []
        contents = f.read().decode()
        for raw in re.split('\s{2,}', contents):
        # X就选择为raw,而Y比X快一个值，最后一个为‘\n’
            x.append(raw)
            x_str = list(raw)
            l.append(len(x_str))

        l = np.asarray(l)
        x = np.asarray(x)
        return l, x

seq_len, X_train= read_txt('./data/data.txt')

def sentences_to_indices(sentence, word_to_index, Len):
    """
    输入的是X（字符串类型的句子数组），再转化为对应的句子列表，
    输出的是能够让Embedding()函数接受的列表或矩阵。

    参数：
        X -- 句子数组，维度为(m, 1)
        word_to_index -- 字典类型的单词到索引的映射
        max_len -- 最大句子的长度，数据集中所有的句子的长度都不会超过它。

    返回：
        X_indices -- 对应于X中的单词索引数组，维度为(m, max_len)
    """
    # 使用0初始化X_indices
    X_indices = np.zeros((Len), dtype=np.float32)
    Y_indices = np.zeros_like(X_indices)
    sentences_words = list(sentence)
    # 初始化j为0
    j = 0
    # 遍历这个单词列表
    for w in sentences_words:
        # 将X_indices的第j号元素为对应的单词索引
        X_indices[j] = word_to_index[w]
        if j > 0:
            Y_indices[j-1] = X_indices[j]
        j += 1
    Y_indices[j-1] = Y_indices[j-2]
    X, Y = torch.LongTensor(X_indices), torch.LongTensor(Y_indices)
    return X, Y