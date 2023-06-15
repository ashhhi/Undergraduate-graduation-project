import tensorflow as tf

class RefineNet:
    def __init__(self, inputs, numFilters=16, droupouts=0.1, doBatchNorm=True):
        self.Model = self.RefineNet(inputs, numFilters, droupouts, doBatchNorm)

    # 定义基本的卷积块
    def Conv2dBlock(self, inputTensor, numFilters, kernelSize=3, doBatchNorm=True):
        x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                                   kernel_initializer='he_normal', padding='same')(inputTensor)
        if doBatchNorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    # 定义RefineNet
    def RefineNet(self, inputImage, numFilters, droupouts, doBatchNorm):
        # 定义Encoder部分
        c1 = self.Conv2dBlock(inputImage, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.Dropout(droupouts)(p1)

        c2 = self.Conv2dBlock(p1, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.Dropout(droupouts)(p2)

        # 定义RefineNet部分
        r1 = self.RefineBlock(p2, numFilters * 4, doBatchNorm=doBatchNorm)
        r2 = self.RefineBlock(r1, numFilters * 4, doBatchNorm=doBatchNorm)
        r3 = self.RefineBlock(r2, numFilters * 4, doBatchNorm=doBatchNorm)

        # 定义Decoder部分
        u4 = tf.keras.layers.Conv2DTranspose(numFilters * 2, (3, 3), strides=(2, 2), padding='same')(r3)
        u4 = tf.keras.layers.concatenate([u4, c2])
        u4 = tf.keras.layers.Dropout(droupouts)(u4)
        c4 = self.Conv2dBlock(u4, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)

        u5 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(c4)
        u5 = tf.keras.layers.concatenate([u5, c1])
        u5 = tf.keras.layers.Dropout(droupouts)(u5)
        c5 = self.Conv2dBlock(u5, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)

        output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
        model = tf.keras.Model(inputs=[inputImage], outputs=[output])
        return model

    # 定义RefineNet中的Refine块
    def RefineBlock(self, inputTensor, numFilters, doBatchNorm=True):
        conv1 = self.Conv2dBlock(inputTensor, numFilters, kernelSize=3, doBatchNorm=doBatchNorm)
        conv2 = self.Conv2dBlock(conv1, numFilters, kernelSize=3, doBatchNorm=doBatchNorm)
        output = tf.keras.layers.Add()([inputTensor, conv2])
        return output