import torch
import pandas as pd
import numpy as np
import pickle
import torchvision.models as models
from transformers import BertTokenizer, AlbertForMaskedLM
import torch
import copy
from torch.nn.functional import softmax


class AlbertCorrector:
    def __init__(self, topK=10, dim=50, pretrained='voidful/albert_chinese_tiny'):
        self.topK = topK
        self.M = torch.load(f'./shape2vec.{dim}d')['shape2vec']  # shape2vec
        self.id2char = get_label_dict()
        self.char2id = {x: y for x, y in zip(self.id2char.values(), self.id2char.keys())}

        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = AlbertForMaskedLM.from_pretrained(pretrained)

    def correctAll(self, sent="", sim_threshold=0.5):
        assert (len(sent) > 1)
        for i in range(len(sent)):
            msk_char = sent[i]
            msk_sent = sent[:i] + "[MASK]" + sent[i + 1:]
            if msk_sent is not None:
                maskpos = self.tokenizer.encode(msk_sent, add_special_tokens=True).index(103)

                input_ids = torch.tensor(self.tokenizer.encode(msk_sent, add_special_tokens=True)).unsqueeze(
                    0)  # Batch size 1
                outputs = self.model(input_ids, masked_lm_labels=input_ids)

                loss, prediction_scores = outputs[:2]

                # logit_prob = softmax(prediction_scores[0, maskpos], dim=0).data.tolist()

                _, indices = torch.topk(prediction_scores[0, maskpos], k=self.topK, dim=0)

                for idx in indices:
                    idx = idx.item()
                    predicted_token = self.tokenizer.convert_ids_to_tokens([idx])[0]
                    sim = AlbertCorrector.charSim(self, c1=msk_char, c2=predicted_token)
                    if sim is not None and sim > sim_threshold:
                        if sent[i] != predicted_token:
                            print(f"{sent[i]} -> {predicted_token}")
                        sent = sent[:i] + predicted_token + sent[i + 1:]
                        break
        return sent

    def charSim(self, c1="于", c2="干"):
        # if not exists
        if self.char2id.get(c1) is None or self.char2id.get(c2) is None:
            return
        sim = torch.cosine_similarity(self.M[self.char2id[c1]], self.M[self.char2id[c2]], dim=-1)
        return sim


def exShapeMatrix(m_path='ShapeNet.pth', d=300):
    m_path = m_path.replace('Net', 'Net_'+str(d)+'d')
    pretrained_dict = torch.load(m_path)
    M = pretrained_dict['model']['classifier.0.weight']
    # resave
    torch.save({'shape2vec': M}, f'shape2vec.{d}d')
    print("Saved!")


def get_label_dict():
    f = open('./chinese_labels', 'rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict


if __name__ == "__main__":
    corrector = AlbertCorrector()
    result = corrector.correctAll("拜登拟任命亚州事务王管")
    print(result)
