import torch
import numpy as np
import pickle
from transformers import BertTokenizer, AlbertForMaskedLM
import torch
from torch.nn.functional import softmax
from albert_LM import AlbertLM

class ViterbiCorrector:
    def __init__(self, topK=10, dim=50, pretrained='voidful/albert_chinese_tiny'):
        self.topK = topK
        self.M = torch.load(f'./shape2vec.{dim}d')['shape2vec']  # shape2vec
        self.id2char = get_label_dict()
        self.char2id = {x: y for x, y in zip(self.id2char.values(), self.id2char.keys())}

        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.LM = AlbertLM()

    def correctAll(self, candidate=None, sim_threshold=0.5):
        p = np.array(candidate)
        path = []
        sent = ""
        vp = 1

        for i in range(0, p.shape[0]):

            max_p = 0
            max_p_idx = 0
            for j in range(p.shape[1]):
                prob1 = 10 * self.LM.getNextWordProb(pre_sent=sent, next_word=p[i][j][1], make_log=False)
                if len(sent) == 0:
                    prob2 = 0
                else:
                    prob2 = 10 * self.LM.getNextWordProb(pre_sent=sent[-1], next_word=p[i][j][1], make_log=False)

                prob_ocr = float(p[i][j][0])
                prob = (prob1 + prob2) * vp * prob_ocr
                if prob > max_p:
                    max_p = prob
                    max_p_idx = j
            vp = max_p
            sent += p[i][max_p_idx][1]
            path.append(max_p_idx)
        return sent





def get_label_dict():
    f = open('./chinese_labels', 'rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict


if __name__ == "__main__":
    p = [[[0.996525228023529, '要'], [0.0009228861308656633, '．'], [0.0006796037196181715, '（'],
          [0.0005814400501549244, '、'], [0.0002970475470647216, '0']],
         [[0.9986881613731384, '统'], [0.0003068553050979972, '结'], [0.0002538849657867104, '纺'],
          [0.00021488533820956945, '筑'], [0.00019387481734156609, '洗']],
         [[0.7405536770820618, '舞'], [0.19035309553146362, '筹'], [0.030714882537722588, '队'],
          [0.004747191444039345, '取'], [0.0027697915211319923, '职']],
         [[0.9999978542327881, '好'], [1.0926590903181932e-06, '奸'], [7.031681548141933e-08, '"'],
          [6.275376307485203e-08, '柱'], [5.675562064766382e-08, '仔']],
         [[0.9943169951438904, '发'], [0.005682302638888359, '友'], [1.9872354428684957e-08, '未'],
          [1.700920648772808e-08, '投'], [1.700920648772808e-08, 'd']],
         [[0.9999834299087524, '展'], [6.307650892267702e-06, '辰'], [2.448814257149934e-06, '屈'],
          [1.8715857095230604e-06, '届'], [1.2366557484710938e-06, '屡']],
         [[0.9999990463256836, '和'], [6.364161322380824e-07, '那'], [1.295288001301742e-07, '而'],
          [5.625532750741513e-08, '邢'], [2.429750267651798e-08, '郁']],
         [[0.9999954700469971, '安'], [1.635241687836242e-06, '字'], [1.5394407455460168e-06, '主'],
          [8.23535003746656e-07, '空'], [2.0006595491395274e-07, '宏']],
         [[0.9998202323913574, '全'], [0.0001681729918345809, '伞'], [5.7796646615315694e-06, '金'],
          [3.76805110136047e-06, '4'], [4.1523713889546343e-07, '坐']]]

    corrector = ViterbiCorrector()
    result = corrector.correctAll(candidate=p)
    print(result)
