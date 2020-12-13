import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import OCR.lib.utils.utils as utils
import OCR.lib.models.crnn as crnn
import OCR.lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='OCR/lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='OCR/images/for_ocr.png', help='the path to image')
    parser.add_argument('--checkpoint', type=str, default='OCR/output/checkpoints/mixed_second_finetune_acc_97P7.pth',
                        help='the path to checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args


def recognition(config, img, model, converter, device):
    h, w = img.shape
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    preds1 = preds.squeeze()
    #print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    #softmax = torch.index_select(preds1,0,preds[preds > 0])    #softmax get
    index = torch.nonzero(torch.gt(preds,torch.tensor([0]).cuda())).squeeze()
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    log_softmax = torch.index_select(preds1.T,1,index).T
    new_log,index = torch.sort(log_softmax,dim = -1,descending=True)
    new_log = torch.exp(new_log[:,:20])
    index = index[:,:20]
    matrix = []
    for i in range(log_softmax.shape[0]):
        str = converter.decode(index[i].data, Variable(torch.IntTensor([index[i].shape[0]])).data, raw=False)
        new_log_ = new_log[i].cpu().detach().numpy().tolist()
        count = 0
        temp = []
        for j in range(index.shape[1] - 1):
            if str[j] >= 'a' and str[j] <= 'z' or str[j] >= 'A' and str[j] <= 'Z':
                continue
            else:
                count+= 1
                temp.append([new_log_[j],str[j]])
                if count >= 5:
                    break
        if len(temp) < 5:
            [temp.append([1e-10,"#"]) for i in range(5 - len(temp))]

        #print(temp)
        matrix.append(temp)

    #print('results: {0}'.format(sim_pred),"\n",matrix)
    return sim_pred,matrix

def OCR_OR_LOGMAX(addition = None,fileName = None):
    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    img = cv2.imread(fileName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    result,matrix = recognition(config, img, model, converter, device)
    #print(result)
    if addition != None:
        return result,matrix
    else:
        return result
