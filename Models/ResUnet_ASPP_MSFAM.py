import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalMaxPool2D, Dense, Activation, GlobalAvgPool2D, UpSampling2D, BatchNormalization,DepthwiseConv2D



class ResUnet_ASPP_MSFAM:
    def __init__(self, inputs, numFilters=16, droupouts=0.1):
        self.Model = self.Unet(inputs, numFilters, droupouts)

    def MSFAM(self, input, numFilters, ratio=16):
        part1 = GlobalAvgPool2D()(input)
        part1 = tf.reshape(part1, [-1, 1, 1, numFilters])
        part1 = Conv2D(filters=numFilters / ratio, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(part1)
        part1 = BatchNormalization()(part1)
        part1 = Activation('relu')(part1)
        part1 = Conv2D(filters=numFilters, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(part1)
        part1 = BatchNormalization()(part1)

        part2 = GlobalMaxPool2D()(input)
        part2 = tf.reshape(part2, [-1, 1, 1, numFilters])
        part2 = Conv2D(filters=numFilters / ratio, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(part2)
        part2 = BatchNormalization()(part2)
        part2 = Activation('relu')(part2)
        part2 = Conv2D(filters=numFilters, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(part2)
        part2 = BatchNormalization()(part2)

        part3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False)(input)
        part3 = BatchNormalization()(part3)
        part3 = Activation('relu')(part3)
        part3 = Conv2D(filters=numFilters, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(part3)
        part3 = BatchNormalization()(part3)

        res = Activation('sigmoid')(part1 + part2 + part3)
        res = tf.multiply(res, input)  # 点乘
        return res

    def ResNetBlock(self, inputTensor, numFilters, kernelSize=3, residual_path=True):

        # first Conv
        x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                                   kernel_initializer='he_normal', padding='same')(inputTensor)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Second Conv
        x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                                   kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        residual = inputTensor
        if residual_path:
            residual = Conv2D(filters=numFilters,kernel_size=(1,1),strides=1,padding='same',use_bias=False)(residual)
            residual = tf.keras.layers.BatchNormalization()(residual)
        x = tf.keras.layers.Activation('relu')(x+residual)

        return x


    def ASPP(self, input, numfilters):
        filter1 = tf.constant(value=1, shape=[1, 1, numfilters, numfilters], dtype=tf.float32)
        filter2 = tf.constant(value=1, shape=[3, 3, numfilters, numfilters], dtype=tf.float32)
        a1 = tf.nn.atrous_conv2d(input, filter1, rate=1, padding='SAME')
        a2 = tf.nn.atrous_conv2d(input, filter2, rate=3, padding='SAME')
        a3 = tf.nn.atrous_conv2d(input, filter2, rate=12, padding='SAME')
        a4 = tf.nn.atrous_conv2d(input, filter2, rate=18, padding='SAME')

        a5 = GlobalAvgPool2D()(input)
        size = input.shape[1:3]
        a5 = tf.reshape(a5, [-1, 1, 1, numfilters])
        a5 = UpSampling2D(size=size,interpolation="bilinear")(a5)

        x = tf.keras.layers.concatenate([a1, a2, a3, a4, a5])
        y = Conv2D(filters=numfilters, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x)
        return y

    # Now defining Unet
    def Unet(self, inputImage, numFilters, droupouts):
        # defining encoder Path
        c1 = self.ResNetBlock(inputImage, numFilters * 1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.Dropout(droupouts)(p1)

        c2 = self.ResNetBlock(p1, numFilters * 2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.Dropout(droupouts)(p2)

        c3 = self.ResNetBlock(p2, numFilters * 4)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        p3 = tf.keras.layers.Dropout(droupouts)(p3)

        c4 = self.ResNetBlock(p3, numFilters * 8)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
        p4 = tf.keras.layers.Dropout(droupouts)(p4)

        c5 = self.ResNetBlock(p4, numFilters * 16)
        a = self.ASPP(c5, numFilters * 16)

        # defining decoder path
        u6 = tf.keras.layers.Conv2DTranspose(numFilters * 8, (3, 3), strides=(2, 2), padding='same')(a)
        a1 = self.MSFAM(c4, numFilters * 8)
        u6 = tf.keras.layers.concatenate([u6, a1])
        u6 = tf.keras.layers.Dropout(droupouts)(u6)
        c6 = self.ResNetBlock(u6, numFilters * 8)

        u7 = tf.keras.layers.Conv2DTranspose(numFilters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        a2 = self.MSFAM(c3, numFilters * 4)
        u7 = tf.keras.layers.concatenate([u7, a2])
        u7 = tf.keras.layers.Dropout(droupouts)(u7)
        c7 = self.ResNetBlock(u7, numFilters * 4)

        u8 = tf.keras.layers.Conv2DTranspose(numFilters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        a3 = self.MSFAM(c2, numFilters * 2)
        u8 = tf.keras.layers.concatenate([u8, a3])
        u8 = tf.keras.layers.Dropout(droupouts)(u8)
        c8 = self.ResNetBlock(u8, numFilters * 2)

        u9 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        a4 = self.MSFAM(c1, numFilters * 1)
        u9 = tf.keras.layers.concatenate([u9, a4])
        u9 = tf.keras.layers.Dropout(droupouts)(u9)
        c9 = self.ResNetBlock(u9, numFilters * 1)

        output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = tf.keras.Model(inputs=[inputImage], outputs=[output])
        return model
