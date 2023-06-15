# https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def LoadData(frameObj=None, imgPath=None, maskPath=None, shape=128):
    imgNames = os.listdir(imgPath)
    maskNames = []

    ## generating mask names
    for mem in imgNames:
        mem = mem.split('_')[0]
        if mem not in maskNames:
            maskNames.append(mem)

    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'

    for i in tqdm(range(len(imgNames))):
        try:
            img = plt.imread(imgAddr + maskNames[i] + '_sat.jpg')
            mask = plt.imread(maskAddr + maskNames[i] + '_mask.png')

        except:
            continue
        img = cv2.resize(img, (shape, shape))
        mask = cv2.resize(mask, (shape, shape))
        frameObj['img'].append(img)
        frameObj['mask'].append(mask[:, :, 0])  # this is because its a binary mask and img is present in channel 0

    return frameObj
#
#
# def LoadData(frameObj=None, imgPath=None, maskPath=None, shape=128):
#     imgNames = os.listdir(imgPath)
#     maskNames = []
#
#     ## generating mask names
#     for mem in imgNames:
#         mem = mem.split('_')[0]
#         if mem not in maskNames:
#             maskNames.append(mem)
#
#     imgAddr = imgPath + '/'
#     maskAddr = maskPath + '/'
#
#     for i in range(len(imgNames)):
#         try:
#             img = imgAddr + maskNames[i] + '_sat.jpg'
#             mask = maskAddr + maskNames[i] + '_mask.png'
#
#         except:
#             continue
#         frameObj['img'].append(img)
#         frameObj['mask'].append(mask)  # this is because its a binary mask and img is present in channel 0
#
#     return frameObj