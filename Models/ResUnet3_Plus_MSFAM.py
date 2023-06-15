import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalMaxPool2D, Dense, Activation, GlobalAvgPool2D, UpSampling2D, BatchNormalization,DepthwiseConv2D


def adaptive_max_pool2d(x, output_size):
    input_shape = x.shape
    input_height = input_shape[1]
    input_width = input_shape[2]

    target_height = output_size[0]
    target_width = output_size[1]

    stride_height = input_height // target_height
    stride_width = input_width // target_width
    pooled = tf.keras.layers.MaxPooling2D((stride_height, stride_width))(x)
    return pooled
class ResUnet3_Plus_MSFAM:
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
        e1 = self.ResNetBlock(inputImage, numFilters * 1, kernelSize=3)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(e1)
        p1 = tf.keras.layers.Dropout(droupouts)(p1)
        e1_1 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                     kernel_initializer='he_normal', padding='same')(e1)
        e1_1 = self.MSFAM(e1_1, numFilters)
        e1_2 = tf.keras.layers.MaxPooling2D((2, 2))(e1)
        e1_2 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(e1_2)
        e1_2 = self.MSFAM(e1_2, numFilters)
        e1_3 = tf.keras.layers.MaxPooling2D((4, 4))(e1)
        e1_3 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                   kernel_initializer='he_normal', padding='same')(e1_3)
        e1_3 = self.MSFAM(e1_3, numFilters)
        e1_4 = tf.keras.layers.MaxPooling2D((8, 8))(e1)
        e1_4 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(e1_4)
        e1_4 = self.MSFAM(e1_4, numFilters)

        e2 = self.ResNetBlock(p1, numFilters * 2, kernelSize=3)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(e2)
        p2 = tf.keras.layers.Dropout(droupouts)(p2)
        e2_2 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                     kernel_initializer='he_normal', padding='same')(e2)
        e2_2 = self.MSFAM(e2_2, numFilters)
        e2_3 = tf.keras.layers.MaxPooling2D((2, 2))(e2)
        e2_3 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                     kernel_initializer='he_normal', padding='same')(e2_3)
        e2_3 = self.MSFAM(e2_3, numFilters)
        e2_4 = tf.keras.layers.MaxPooling2D((4, 4))(e2)
        e2_4 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(e2_4)
        e2_4 = self.MSFAM(e2_4, numFilters)

        e3 = self.ResNetBlock(p2, numFilters * 4, kernelSize=3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(e3)
        p3 = tf.keras.layers.Dropout(droupouts)(p3)
        e3_3 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                     kernel_initializer='he_normal', padding='same')(e3)
        e3_3 = self.MSFAM(e3_3, numFilters)
        e3_4 = tf.keras.layers.MaxPooling2D((2, 2))(e3)
        e3_4 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(e3_4)
        e3_4 = self.MSFAM(e3_4, numFilters)

        e4 = self.ResNetBlock(p3, numFilters * 8, kernelSize=3)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(e4)
        p4 = tf.keras.layers.Dropout(droupouts)(p4)
        e4_4 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                     kernel_initializer='he_normal', padding='same')(e4)
        e4_4 = self.MSFAM(e4_4, numFilters)

        e5 = self.ResNetBlock(p4, numFilters * 16, kernelSize=3)

        # defining decoder path
        d5_1 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(16, 16), padding='same')(e5)
        d5_1 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d5_1)
        d5_1 = self.MSFAM(d5_1, numFilters)
        d5_2 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(8, 8), padding='same')(e5)
        d5_2 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d5_2)
        d5_2 = self.MSFAM(d5_2, numFilters)
        d5_3 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(4, 4), padding='same')(e5)
        d5_3 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d5_3)
        d5_3 = self.MSFAM(d5_3, numFilters)
        d5_4 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(2, 2), padding='same')(e5)
        d5_4 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d5_4)
        d5_4 = self.MSFAM(d5_4, numFilters)

        d4 = tf.keras.layers.concatenate([d5_4, e4_4, e3_4, e2_4, e1_4])
        d4 = tf.keras.layers.Conv2D(filters=numFilters * 5, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d4)
        d4 = tf.keras.layers.BatchNormalization()(d4)
        d4 = Activation('relu')(d4)
        d4 = tf.keras.layers.Dropout(droupouts)(d4)

        d4_3 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(2, 2), padding='same')(d4)
        d4_3 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d4_3)
        d4_3 = self.MSFAM(d4_3, numFilters)
        d4_2 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(4, 4), padding='same')(d4)
        d4_2 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d4_2)
        d4_2 = self.MSFAM(d4_2, numFilters)
        d4_1 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(8, 8), padding='same')(d4)
        d4_1 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d4_1)
        d4_1 = self.MSFAM(d4_1, numFilters)

        d3 = tf.keras.layers.concatenate([d5_3, d4_3, e3_3, e2_3, e1_3])
        d3 = tf.keras.layers.Conv2D(filters=numFilters * 5, kernel_size=(3, 3),
                                    kernel_initializer='he_normal', padding='same')(d3)
        d3 = tf.keras.layers.BatchNormalization()(d3)
        d3 = Activation('relu')(d3)
        d3 = tf.keras.layers.Dropout(droupouts)(d3)
        d3_2 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(2, 2), padding='same')(d3)
        d3_2 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d3_2)
        d3_2 = self.MSFAM(d3_2, numFilters)
        d3_1 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(4, 4), padding='same')(d3)
        d3_1 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d3_1)
        d3_1 = self.MSFAM(d3_1, numFilters)

        d2 = tf.keras.layers.concatenate([d5_2, d4_2, d3_2, e2_2, e1_2])
        d2 = tf.keras.layers.Conv2D(filters=numFilters * 5, kernel_size=(3, 3),
                                    kernel_initializer='he_normal', padding='same')(d2)
        d2 = tf.keras.layers.BatchNormalization()(d2)
        d2 = Activation('relu')(d2)
        d2 = tf.keras.layers.Dropout(droupouts)(d2)
        d2_1 = tf.keras.layers.Conv2DTranspose(numFilters, (3, 3), strides=(2, 2), padding='same')(d2)
        d2_1 = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(3, 3),
                                      kernel_initializer='he_normal', padding='same')(d2_1)
        d2_1 = self.MSFAM(d2_1, numFilters)

        d1 = tf.keras.layers.concatenate([d5_1, d4_1, d3_1, d2_1, e1_1])
        d1 = tf.keras.layers.Conv2D(filters=numFilters * 5, kernel_size=(3, 3),
                                    kernel_initializer='he_normal', padding='same')(d1)
        d1 = tf.keras.layers.BatchNormalization()(d1)
        d1 = Activation('relu')(d1)
        d1 = tf.keras.layers.Dropout(droupouts)(d1)

        # 深监督
        # 分类引导模块
        sort_layer = tf.keras.layers.Dropout(droupouts)(e5)
        sort_layer = tf.keras.layers.Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(sort_layer)
        sort_layer = adaptive_max_pool2d(sort_layer, [1, 1])
        sort_layer = Activation('sigmoid')(sort_layer)

        output1 = tf.keras.layers.Conv2D(1, (3, 3), kernel_initializer='he_normal', padding='same')(d1)
        output1 = tf.multiply(output1, sort_layer)
        output2 = tf.keras.layers.Conv2D(1, (3, 3), kernel_initializer='he_normal', padding='same')(d2)
        output2 = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same')(output2)
        output2 = tf.multiply(output2, sort_layer)
        output3 = tf.keras.layers.Conv2D(1, (3, 3), kernel_initializer='he_normal', padding='same')(d3)
        output3 = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(4, 4), padding='same')(output3)
        output3 = tf.multiply(output3, sort_layer)
        output4 = tf.keras.layers.Conv2D(1, (3, 3), kernel_initializer='he_normal', padding='same')(d4)
        output4 = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(8, 8), padding='same')(output4)
        output4 = tf.multiply(output4, sort_layer)
        output5 = tf.keras.layers.Conv2D(1, (3, 3), kernel_initializer='he_normal', padding='same')(e5)
        output5 = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(16, 16), padding='same')(output5)
        output5 = tf.multiply(output5, sort_layer)


        tmp = [output2, output3, output4, output5, output1]
        outputs = []
        for item in tmp:
            outputs.append(Activation('sigmoid')(item))
        model = tf.keras.Model(inputs=[inputImage], outputs=outputs)
        return model
