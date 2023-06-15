
import os
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from PIL import Image
import matplotlib.image as mpimg
import skimage.io
import torchvision.transforms as transforms
from datetime import datetime


# 评价指标
def sorensen_dices(y_true, y_pred):  # 计算三个评价指标
    y_pred = np.round(y_pred)
    y_true = np.round(y_true)
    intersection = np.sum(y_true * y_pred)
    TN = np.sum((1 - y_true) * (1 - y_pred))
    TP = intersection
    FN = np.sum(y_true * (1 - y_pred))
    FP = np.sum((1 - y_true) * y_pred)

    # Re_y_true = np.ones(y_true.shape)
    # Re_y_true[y_true > 0] = 0
    # Re_y_pred = np.ones(y_pred.shape)
    # Re_y_pred[y_pred > 0] = 0
    # ReS_intse = np.sum(Re_y_true * y_pred) # 真实标注与分割结果的交集区域的像素数量。
    # ReT_intse = np.sum(y_true * Re_y_pred) #
    # if (np.sum(y_true) + np.sum(y_pred)) == 0:
    #     FPD = 0
    #     FND = 0
    #     print('hh')
    # else:
    #     FPD = 2 * ReS_intse / (np.sum(y_true) + np.sum(y_pred))
    #     FND = 2 * ReT_intse / (np.sum(y_true) + np.sum(y_pred))

    FPD = FP / (TN + FP)
    FND = FN / (TP + FN)

    IOU = TP / (TP + FP + FN)

    # DSI
    # DSI = (2 * intersection + 1) / (np.sum(y_true) + np.sum(y_pred) + 1)
    DSI = (2 * TP) / (2 * TP + FP + FN)

    # Jaccard相似度
    x = y_pred + y_true
    union = np.sum(x > 0)
    Js = (intersection + 1) / (union + 1)

    #精确度precision
    precision = (intersection + 1) / (np.sum(y_pred) + 1)

    # 特异性
    specificity = TN / (TN + FP)

    #灵敏度sensitivity
    sensitivity = TP / (TP + FN)

    #正确率
    accuary = (TP + TN) / (TP + TN + FP + FN)

    #f1分数
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)



    return DSI, FPD, FND, Js, precision, specificity, sensitivity,accuary,f1,IOU


# 进行评估，获得评估结果
def Evaluation(Predict_Path, Label_Path, target_size, Begin=0, End=0):  # 输出三个评测指标（整体上）
    DSIs = 0
    FPDs = 0
    FNDs = 0
    Jss = 0
    precisions=0
    specificitys=0
    sensitivitys=0
    accuarys=0
    f1s = 0
    ious = 0
    PA = 0
    sum = End - Begin + 1
    transform1 = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ]
    )

    for k in tqdm(range(End)):
        y_true = Image.open(os.path.join(Label_Path, "%d.png" % k))#0~255 array
        y_true = y_true.convert('L')
        y_true = transform1(y_true)
        zero = torch.zeros_like(y_true)
        one = torch.ones_like(y_true)
        y_true = torch.where(y_true > 0.5, one, zero)
        y_true = y_true.numpy().squeeze()
        y_true = y_true.astype('float32')#(256,256)  0、1
        #y_pred = skimage.io.imread(os.path.join(Predict_Path, "%d.png" % k))#0、255 array  (256,256)

        y_pred = Image.open(os.path.join(Predict_Path, "%d.png" % k))
        y_pred = y_pred.convert('L')
        y_pred = transform1(y_pred)
        zero = torch.zeros_like(y_pred)
        one = torch.ones_like(y_pred)
        y_pred = torch.where(y_pred > 0.5, one, zero)
        y_pred = y_pred.numpy().squeeze()
        y_pred = y_pred.astype('float32')



        e, f, g, j, precision, specificity, sensitivity, accuary, f1,iou = sorensen_dices(y_true, y_pred)
        DSIs = DSIs + e
        FPDs = FPDs + f
        FNDs = FNDs + g
        Jss = Jss + j
        precisions = precisions + precision
        specificitys = specificitys + specificity
        sensitivitys = sensitivitys + sensitivity
        accuarys = accuarys+accuary
        f1s = f1s + f1
        ious = ious+iou
    DSIsm = DSIs / sum
    FPDsm = FPDs / sum
    FNDsm = FNDs / sum
    Jssm = Jss / sum
    precisionsm = precisions / sum
    specificitysm = specificitys / sum
    sensitivitysm = sensitivitys / sum
    accuarysm = accuarys / sum
    f1sm = f1s / sum
    iousm = ious/sum
    output = "%s:\n\nDSI: %f,\nFPD: %f,\nFND: %f,\nprecision: %f,\nJs: %f,\nspecificity: %f,\nrecall: %f,\naccuary: %f,\nf1: %f,\niou: %f" % (datetime.now(), DSIsm, FPDsm, FNDsm,precisionsm, Jssm, specificitysm,sensitivitysm,accuarysm,f1sm,iousm)
    print(output)
    # test_txt = os.path.join(ROOT_DIR, 'test/result/c2/test.txt')
    # os.makedirs(os.path.dirname(test_txt), exist_ok=True)
    # with open(test_txt, "w") as f:
    #     f.write(output + '\n')
    # Js方差
    return

# test()
Evaluation(Predict_Path=r"C:\DataSet\DeepGlobal\ResUnet3Plus_ASPP_CBAMPlus\pred", Label_Path='C:\DataSet\DeepGlobal\ResUnet3Plus_ASPP_CBAMPlus\gt', target_size=(256,256), Begin=1, End=6226)
