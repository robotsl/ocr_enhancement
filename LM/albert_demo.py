# pip install transformers==2.2.2
from transformers import BertTokenizer, AlbertForMaskedLM
import torch
from torch.nn.functional import softmax

pretrained = 'voidful/albert_chinese_tiny'
tokenizer = BertTokenizer.from_pretrained(pretrained)
model = AlbertForMaskedLM.from_pretrained(pretrained)

inputtext = "样[MASK]"
# inputtext = "最常用的场合就是求一个样[MASK]被网络认为前 k 个最可能属于的类别。我们就用这个场景为例，说明函数的使用方法。"

# vocab[103] is [MASK]
maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)

input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, masked_lm_labels=input_ids)

loss, prediction_scores = outputs[:2]

# print(prediction_scores.shape)  # torch.Size([1, 12, 21128]) where 12 = len(text)+[MASK] = 8 + 3
# print(prediction_scores[0, maskpos].shape)  # torch.Size([21128])

logit_prob = softmax(prediction_scores[0, maskpos], dim=0).data.tolist()
# print(len(logit_prob)) # 21128

# predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()

_, indices = torch.topk(prediction_scores[0, maskpos], k=10, dim=0)


for idx in indices:
    idx = idx.item()
    predicted_token = tokenizer.convert_ids_to_tokens([idx])[0]
    print(predicted_token, logit_prob[idx])

# print(predicted_index) # predicted_index = 7553
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# print(predicted_token, logit_prob[predicted_index])
