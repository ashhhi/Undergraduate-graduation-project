import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalMaxPool2D, Dense, Activation, GlobalAvgPool2D

class ResUnet_PlusPlus:
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
        x0_0 = self.ResNetBlock(inputImage, numFilters * 1, kernelSize=3)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(x0_0)
        p1 = tf.keras.layers.Dropout(droupouts)(p1)

        x1_0 = self.ResNetBlock(p1, numFilters * 2, kernelSize=3)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(x1_0)
        p2 = tf.keras.layers.Dropout(droupouts)(p2)

        x2_0 = self.ResNetBlock(p2, numFilters * 4, kernelSize=3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(x2_0)
        p3 = tf.keras.layers.Dropout(droupouts)(p3)

        x3_0 = self.ResNetBlock(p3, numFilters * 8, kernelSize=3)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(x3_0)
        p4 = tf.keras.layers.Dropout(droupouts)(p4)

        x4_0 = self.ResNetBlock(p4, numFilters * 16, kernelSize=3)

        # defining decoder path
        u1_0 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(x1_0)
        x0_1 = tf.keras.layers.concatenate([x0_0, u1_0])
        x0_1 = tf.keras.layers.Dropout(droupouts)(x0_1)
        x0_1 = self.ResNetBlock(x0_1, numFilters * 1, kernelSize=3)

        u2_0 = tf.keras.layers.Conv2DTranspose(numFilters * 2, (3, 3), strides=(2, 2), padding='same')(x2_0)
        x1_1 = tf.keras.layers.concatenate([x1_0, u2_0])
        x1_1 = tf.keras.layers.Dropout(droupouts)(x1_1)
        x1_1 = self.ResNetBlock(x1_1, numFilters * 2, kernelSize=3)

        u3_0 = tf.keras.layers.Conv2DTranspose(numFilters * 4, (3, 3), strides=(2, 2), padding='same')(x3_0)
        x2_1 = tf.keras.layers.concatenate([x2_0, u3_0])
        x2_1 = tf.keras.layers.Dropout(droupouts)(x2_1)
        x2_1 = self.ResNetBlock(x2_1, numFilters * 4, kernelSize=3)

        u4_0 = tf.keras.layers.Conv2DTranspose(numFilters * 8, (3, 3), strides=(2, 2), padding='same')(x4_0)
        x3_1 = tf.keras.layers.concatenate([x3_0, u4_0])
        x3_1 = tf.keras.layers.Dropout(droupouts)(x3_1)
        x3_1 = self.ResNetBlock(x3_1, numFilters * 8, kernelSize=3)

        u1_1 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(x1_1)
        x0_2 = tf.keras.layers.concatenate([x0_0, x0_1, u1_1])
        x0_2 = tf.keras.layers.Dropout(droupouts)(x0_2)
        x0_2 = self.ResNetBlock(x0_2, numFilters * 1, kernelSize=3)

        u2_1 = tf.keras.layers.Conv2DTranspose(numFilters * 2, (3, 3), strides=(2, 2), padding='same')(x2_1)
        x1_2 = tf.keras.layers.concatenate([x1_0, x1_1, u2_1])
        x1_2 = tf.keras.layers.Dropout(droupouts)(x1_2)
        x1_2 = self.ResNetBlock(x1_2, numFilters * 2, kernelSize=3)

        u3_1 = tf.keras.layers.Conv2DTranspose(numFilters * 4, (3, 3), strides=(2, 2), padding='same')(x3_1)
        x2_2 = tf.keras.layers.concatenate([x2_0, x2_1, u3_1])
        x2_2 = tf.keras.layers.Dropout(droupouts)(x2_2)
        x2_2 = self.ResNetBlock(x2_2, numFilters * 4, kernelSize=3)

        u1_2 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(x1_2)
        x0_3 = tf.keras.layers.concatenate([x0_0, x0_1, x0_2, u1_2])
        x0_3 = tf.keras.layers.Dropout(droupouts)(x0_3)
        x0_3 = self.ResNetBlock(x0_3, numFilters * 1, kernelSize=3)

        u2_2 = tf.keras.layers.Conv2DTranspose(numFilters * 2, (3, 3), strides=(2, 2), padding='same')(x2_2)
        x1_3 = tf.keras.layers.concatenate([x1_0, x1_1, x1_2, u2_2])
        x1_3 = tf.keras.layers.Dropout(droupouts)(x1_3)
        x1_3 = self.ResNetBlock(x1_3, numFilters * 2, kernelSize=3)

        u1_3 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(x1_3)
        x0_4 = tf.keras.layers.concatenate([x0_0, x0_1, x0_2, x0_3, u1_3])
        x0_4 = tf.keras.layers.Dropout(droupouts)(x0_4)
        x0_4 = self.ResNetBlock(x0_4, numFilters * 1, kernelSize=3)

        # Deep Supervision
        output1 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x0_1)
        output2 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x0_2)
        output3 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x0_3)
        output4 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x0_4)
        print(output1.shape)
        print(output2.shape)
        print(output3.shape)
        print(output4.shape)
        model = tf.keras.Model(inputs=[inputImage], outputs=[output1,output2,output3,output4])
        return model
