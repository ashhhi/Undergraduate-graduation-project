import os



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import numpy as np
import re
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import json

# defining function for dataLoading function
from DataLoader.DeepGlobalRoad import LoadData
from Models.Unet import Unet
from Models.Unet_CBAM import Unet_CBAM
from Models.ResUnet import ResUnet
from Models.ResUnet_CBAM import ResUnet_CBAM
from Models.ResUnet_MSFAM import ResUnet_MSFAM
from Models.ResUnet_ASPP_CBAM import ResUnet_ASPP_CBAM
from Models.ResUnet_ASPP_MSFAM import ResUnet_ASPP_MSFAM
from Models.ResUnet_PlusPlus import ResUnet_PlusPlus
from Models.ResUnet_PlusPlus_CBAM import ResUnet_PlusPlus_CBAM
from Models.ResUnet3_Plus import ResUnet3_Plus
from Models.ResUnet3_Plus_MSFAM import ResUnet3_Plus_MSFAM
from Models.ResUnet3Plus_CBAMPlus import ResUnet3Plus_CBAMPlus
from Models.ResUnet3_Plus_CBAM import ResUnet3_Plus_CBAM
from Models.ResUnet3Plus_CBAMPlus_ASPP import ResUnet3Plus_CBAMPlus_ASPP

from Models.FCN_8s import FCN_8s
from Models.DeepLabv3Plus import DeepLabv3Plus
import loss

if __name__ == '__main__':
    print(tf.__version__)
    framObjTrain = {'img': [],
                    'mask': []
                    }
    ## instanctiating model
    inputs = tf.keras.layers.Input((128, 128, 3))
    Net = ResUnet3Plus_CBAMPlus_ASPP(inputs).Model
    Net.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    # Net.compile(optimizer='Adam', loss=loss.lovasz_softmax_loss, metrics=['accuracy'])

    print("Load Data...")
    framObjTrain = LoadData(framObjTrain, imgPath = r'C:\DataSet\DeepGlobal/train',
                            maskPath = r'C:\DataSet\DeepGlobal/train'
                             , shape = 128)
    print("Load Data Successfully")

    retVal = Net.fit(np.array(framObjTrain['img']), np.array(framObjTrain['mask']), epochs=200, verbose=1, batch_size=8)
    Net.save('Outputs/SavedModel/ResUnet.h5')
    with open('history.txt', 'w') as f:
        f.write(str(retVal.history))
    Net.summary()