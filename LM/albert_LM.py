import torch
import pandas as pd
import numpy as np
import pickle
import torchvision.models as models
from transformers import BertTokenizer, AlbertForMaskedLM
import torch
import copy
from torch.nn.functional import softmax


class AlbertLM:
    def __init__(self, pretrained='voidful/albert_chinese_tiny'):

        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = AlbertForMaskedLM.from_pretrained(pretrained)

    def getNextWordProb(self, pre_sent, next_word):
        assert isinstance(pre_sent, str)
        assert isinstance(next_word, str)
        msk_sent = pre_sent + "[MASK]"
        maskpos = self.tokenizer.encode(msk_sent, add_special_tokens=True).index(103)

        input_ids = torch.tensor(self.tokenizer.encode(msk_sent, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids, masked_lm_labels=input_ids)

        loss, prediction_scores = outputs[:2]

        logit_prob = softmax(prediction_scores[0, maskpos], dim=0).data.tolist()

        idx = self.tokenizer.convert_tokens_to_ids(next_word)
        return logit_prob[idx]

    @staticmethod
    def get_label_dict():
        f = open('./chinese_labels', 'rb')
        label_dict = pickle.load(f)
        f.close()
        return label_dict


if __name__ == "__main__":
    LM = AlbertLM()
    prob = LM.getNextWordProb(pre_sent="拜登拟任命亚", next_word="洲")
    print(prob)
