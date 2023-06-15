import tensorflow as tf


class FCN_8s:
    def __init__(self, inputs):
        self.Model = self.Unet(inputs)

    # defining Conv2d block for our u-net
    # this block essentially performs 2 convolution
    def conv_block(self, inputs, filters, kernel_size=3, strides=1, padding='same', activation='relu'):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        return x

    def upsample_block(self, inputs, filters, kernel_size=3, strides=2, padding='same'):
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(inputs)
        return x
    # Now defining Unet
    def Unet(self, inputs):
        # Encoder
        conv1 = self.conv_block(inputs, 32)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv_block(pool1, 64)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_block(pool2, 128)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv_block(pool3, 256)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.conv_block(pool4, 512)
        pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)

        # FCN-8s specific layers
        fc6 = self.conv_block(pool5, 2048)
        drop6 = tf.keras.layers.Dropout(0.5)(fc6)

        fc7 = self.conv_block(drop6, 2048)
        drop7 = tf.keras.layers.Dropout(0.5)(fc7)

        # Decoder
        score_fr = tf.keras.layers.Conv2D(1, (1, 1), activation='relu')(drop7)

        upscore2 = self.upsample_block(score_fr, 1, kernel_size=4, strides=2)
        score_pool5 = tf.keras.layers.Conv2D(1, (1, 1), activation='relu')(conv5)
        fuse_pool5 = tf.keras.layers.Add()([upscore2, score_pool5])

        upscore4 = self.upsample_block(fuse_pool5, 1, kernel_size=4, strides=2)
        score_pool3 = tf.keras.layers.Conv2D(1, (1, 1), activation='relu')(conv4)
        fuse_pool4 = tf.keras.layers.Add()([upscore4, score_pool3])

        upscore8 = self.upsample_block(fuse_pool4, 1, kernel_size=16, strides=8)

        # Output
        output = tf.keras.layers.Activation('sigmoid')(upscore8)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model
