import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from DataLoader.DeepGlobalRoad import LoadData
from tqdm import tqdm

framObjTrain = {'img': [],
                'mask': [],
                }


def predict(valMap, model, num=None):
    ## getting and proccessing val data
    if num:
        img = valMap['img'][0:num]
        mask = valMap['mask'][0:num]
    else:
        img = valMap['img']
        mask = valMap['mask']
    imgProc = np.array(img)

    predictions = model.predict(imgProc)

    return predictions, imgProc, mask


def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(9, 9))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Aerial image')

    plt.subplot(1, 3, 2)
    plt.imshow(predMask)
    plt.title('Predicted Routes')

    plt.subplot(1, 3, 3)
    plt.imshow(groundTruth)
    plt.title('Actual Routes')
    plt.show()


if __name__ == '__main__':
    print("Load Data...")

    framObjTrain = LoadData(framObjTrain, imgPath=r'C:\DataSet\DeepGlobal\train',
                            maskPath=r'C:\DataSet\DeepGlobal\train'
                            , shape=128)
    print("Load Model...")
    unet = tf.keras.models.load_model('Outputs/SavedModel/DeepLabv3Plus.h5')

    total = len(framObjTrain['img'])
    assert total != 0, 'len(framObjTrain[\'img\'] = 0)'
    print("Start Predict")
    sixteenPrediction, actuals, masks = predict(framObjTrain, unet)

    rootDIR = r'C:\DataSet\DeepGlobal\output'
    pred_path = os.path.join(rootDIR, 'pred/')
    gt_path = os.path.join(rootDIR, 'gt/')
    img_path = os.path.join(rootDIR, 'img/')
    if os.path.exists(pred_path) is False:
        os.makedirs(pred_path)
    if os.path.exists(gt_path) is False:
        os.makedirs(gt_path)
    if os.path.exists(img_path) is False:
        os.makedirs(img_path)
    for i in tqdm(range(total)):
        # Plotter(actuals[i], sixteenPrediction[i][:,:,0], masks[i])
        cv2.imwrite(os.path.join(img_path, f'{i}.png'), actuals[i][:, :, 0] * 255)
        cv2.imwrite(os.path.join(pred_path, f'{i}.png'), sixteenPrediction[i][:, :, 0] * 255)
        cv2.imwrite(os.path.join(gt_path, f'{i}.png'), masks[i] * 255)
