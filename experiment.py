from PIL import Image,ImageFont,ImageDraw
import os
import OCR.OCR as OCR
import pandas as pd
import Corrector.albert_corrector as CO
import Corrector.viterbi_corrector as VC

co = CO.AlbertCorrector()
vc = VC.ViterbiCorrector()

def get_data():
    df = pd.read_table("pairs_test.txt",header=None)
    df.columns = ["sample","label"]
    sentences = []
    labels = []
    for i in df["sample"]:
        sentences.append(i)

    for i in df["label"]:
        labels.append(i)

    return sentences,labels

sentences,labels = get_data()
def generate_imgs():
    print("generating imgs...")
    for i in range(len(labels)):
        tmp = sentences[i]
        im = Image.new("RGB", (1700, 73), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype(os.path.join("fonts", "msyh.ttf"),45)
        dr.text((10, 5), tmp, font=font, fill="#000000")  #[10,10]
        im.save('images/{index}.png'.format(index = i))

generate_imgs()

def ocr(sentences = sentences,labels = labels):
    data = []
    vary = []
    matrix = []
    for i in range(len(labels)):
        tmp = sentences[i]
        l_ = labels[i]
        string,matrix_ = OCR.OCR_OR_LOGMAX(addition = 'log',fileName='images/{index}.png'.format(index = i))

        wrong = []
        cur = len(l_) - len(string)
        if cur > 0:
            for j in range(len(l_)):
                if j < len(string) and l_[j] == string[j]:
                    continue
                elif  j < len(string) and  l_[j]!= string[j]:
                    if j + 1 < len(string) and l_[j] == string[j + 1]:
                        string = list(string)
                        string.insert(j,'0')
                        string = ''.join(string)
                    else:
                        pass
                else:
                    pass
                if len(string) == len(l_):
                    break
        elif cur < 0:
            for j in range(len(l_)):
                if j < len(l_) and j < len(string) and l_[j] == string[j]:
                    continue
                elif j < len(l_) and j < len(string) and  l_[j]!= string[j]:
                    if j < len(l_) and j + 1 < len(string)  and l_[j] == string[j + 1]:
                        string = string.replace(string[j],'')
                    elif j < len(l_) and j + 2 < len(string) and l_[j] == string[j + 2]:
                            string = string.replace(string[j], '')
                    else:
                        pass
                else:
                    pass
                if len(string) == len(l_):
                    break
        else:
            pass

        if len(l_) > len(string):
            for j in range(len(l_) - len(string)):
                string += '0'
        if len(l_) < len(string):
            dec = len(string) - len(l_)
            for j in range(dec + 1):
                string.replace(string[len(string) - 1],'')

        print(len(string),len(l_))

        for j in range(len(l_)):
            if tmp[j] != l_[j]:
                wrong.append(j)
        vary.append(wrong)
        #print(vary)
        data.append(string)
        matrix.append(matrix_)

    return data,vary,matrix

data,vary,matrix = ocr(sentences,labels)

def validate_bert():
    '''
    result 是纠正后结果，label_是ground truth, data_是OCR识别的结果,total是字的总和
    TP,TN 分别是本来正确结果认为是正确和错误的次数（data_和result）
    FP,FN 分别是本来错误结果认为是正确和认为错误的次数（label_和result）
    '''
    total = 0
    FN = TN = FP = TP = 0
    ACC = P = R = 0

    #sentences,labels = get_data()
    #data,vary,_ = ocr(sentences,labels)

    for i in range(len(data)):
        # print("=",end='')
        result = co.correctAll(data[i])
        data_ = data[i]
        label_ = labels[i]
        for j in range(min(len(label_), len(data_), len(result))):
            if label_[j] not in "，,。.？?、\\()（）！!;；" or data_[j] not in "，,。.？?、\\()（）！!;；" or result[
                j] not in "，,。.？?、\\()（）！!;；":
                if j in vary[i]:  # 求TN和FP
                    # print("in",data_[j],result[j],label_[j])
                    if result[j] == label_[j] and data_[j] != label_[j]:
                        TN += 1
                    if result[j] == data_[j] and data_[j] != label_[j]:
                        FP += 1
                else:  # 求FN和TP
                    # print("not in",data_[j],result[j],label_[j])
                    if data_[j] == label_[j] and result[j] == label_[j]:
                        TP += 1
                    elif data_[j] == label_[j] and result[j] != label_[j]:
                        FN += 1
            else:
                pass
    print()
    total = TP + TN + FP + FN
    print("total,TP,TN,FP,FN:",total,TP,TN,FP,FN)
    ACC = (TP + TN) / total
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print("ACC,P,R:",ACC,P,R)

def validate_DP():
    '''
    result 是纠正后结果，label_是ground truth, data_是OCR识别的结果,total是字的总和
    TP,TN 分别是本来正确结果认为是正确和错误的次数（data_和result）
    FP,FN 分别是本来错误结果认为是正确和认为错误的次数（label_和result）
    '''
    total = 0
    FN = TN = FP = TP = 0
    ACC = P = R = 0

    #sentences,labels = get_data()
    #data,vary,matrix = ocr(sentences,labels)

    for i in range(len(data)):
        #print("=",end='')
        result = vc.correctAll(matrix[i])
        data_ = data[i]
        label_ = labels[i]
        for j in range(min(len(label_),len(data_),len(result))):
            if label_[j] not in "，,。.？?、\\()（）！!;；" or data_[j] not in "，,。.？?、\\()（）！!;；" or result[j] not in "，,。.？?、\\()（）！!;；":
                if j in vary[i]: #求TN和FP
                    #print("in",data_[j],result[j],label_[j])
                    if result[j] == label_[j] and data_[j] != label_[j]:
                        TN += 1
                    if result[j] == data_[j] and data_[j] != label_[j]:
                        FP += 1
                else:  #求FN和TP
                    #print("not in",data_[j],result[j],label_[j])
                    if data_[j] == label_[j] and result[j] == label_[j]:
                        TP += 1
                    elif data_[j] == label_[j] and result[j] != label_[j]:
                        FN += 1
            else:
                pass
    total = TP + TN + FP + FN
    print("total,TP,TN,FP,FN:",total,TP,TN,FP,FN)
    ACC = (TP + TN) / total
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print("ACC,P,R:",ACC,P,R)


validate_bert()
validate_DP()
