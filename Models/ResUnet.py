import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalMaxPool2D, Dense, Activation, GlobalAvgPool2D

class ResUnet:
    def __init__(self, inputs, numFilters=16, droupouts=0.1):
        self.Model = self.Unet(inputs, numFilters, droupouts)

    # defining Conv2d block for our u-net
    # this block essentially performs 2 convolution
    def ResNetBlock(self, inputTensor, numFilters, kernelSize=3, residual_path=True):

        # first Conv
        x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                                   kernel_initializer='he_normal', padding='same')(inputTensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Second Conv
        x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                                   kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation('relu')(x)

        residual = inputTensor
        if residual_path:
            residual = Conv2D(filters=numFilters,kernel_size=(1,1),strides=1,padding='same',use_bias=False)(residual)
            residual = tf.keras.layers.BatchNormalization()(residual)
        x = tf.keras.layers.Activation('relu')(x+residual)

        return x

    # Now defining Unet
    def Unet(self, inputImage, numFilters, droupouts):
        # defining encoder Path
        c1 = self.ResNetBlock(inputImage, numFilters * 1, kernelSize=3)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.Dropout(droupouts)(p1)

        c2 = self.ResNetBlock(p1, numFilters * 2, kernelSize=3)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.Dropout(droupouts)(p2)

        c3 = self.ResNetBlock(p2, numFilters * 4, kernelSize=3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        p3 = tf.keras.layers.Dropout(droupouts)(p3)

        c4 = self.ResNetBlock(p3, numFilters * 8, kernelSize=3)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
        p4 = tf.keras.layers.Dropout(droupouts)(p4)

        c5 = self.ResNetBlock(p4, numFilters * 16, kernelSize=3)

        # defining decoder path
        u6 = tf.keras.layers.Conv2DTranspose(numFilters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        u6 = tf.keras.layers.Dropout(droupouts)(u6)
        c6 = self.ResNetBlock(u6, numFilters * 8, kernelSize=3)

        u7 = tf.keras.layers.Conv2DTranspose(numFilters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        u7 = tf.keras.layers.Dropout(droupouts)(u7)
        c7 = self.ResNetBlock(u7, numFilters * 4, kernelSize=3)

        u8 = tf.keras.layers.Conv2DTranspose(numFilters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        u8 = tf.keras.layers.Dropout(droupouts)(u8)
        c8 = self.ResNetBlock(u8, numFilters * 2, kernelSize=3)

        u9 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1])
        u9 = tf.keras.layers.Dropout(droupouts)(u9)
        c9 = self.ResNetBlock(u9, numFilters * 1, kernelSize=3)

        output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = tf.keras.Model(inputs=[inputImage], outputs=[output])
        return model
