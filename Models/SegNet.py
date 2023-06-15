import tensorflow as tf

class SegNet:
    def __init__(self, inputs):
        self.Model = self.build_model(inputs)

    def conv_block(self, inputs, filters, kernel_size=3, strides=1, padding='same', activation='relu'):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        return x

    def unpool(self, inputs, pool_indices, output_shape):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]

        pool_indices = tf.cast(pool_indices, tf.int32)
        updates = tf.reshape(inputs, [-1])
        indices = tf.unravel_index(pool_indices, [batch_size, height, width, channels])
        indices = [tf.reshape(index, [-1, 1]) for index in indices]

        indices = tf.concat(indices, axis=-1)
        updates = tf.reshape(updates, [-1])
        return tf.scatter_nd(indices, updates, output_shape)

    def build_model(self, inputs):
        # Encoder
        conv1 = self.conv_block(inputs, 16)
        conv2 = self.conv_block(conv1, 16)
        pool1, pool_indices1 = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv3 = self.conv_block(pool1, 32)
        conv4 = self.conv_block(conv3, 32)
        pool2, pool_indices2 = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv5 = self.conv_block(pool2, 64)
        conv6 = self.conv_block(conv5, 64)
        conv7 = self.conv_block(conv6, 64)
        pool3, pool_indices3 = tf.nn.max_pool_with_argmax(conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv8 = self.conv_block(pool3, 128)
        conv9 = self.conv_block(conv8, 128)
        conv10 = self.conv_block(conv9, 128)
        pool4, pool_indices4 = tf.nn.max_pool_with_argmax(conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv11 = self.conv_block(pool4, 256)
        conv12 = self.conv_block(conv11, 256)
        conv13 = self.conv_block(conv12, 256)
        pool5, pool_indices5 = tf.nn.max_pool_with_argmax(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Decoder
        deconv5 = self.conv_block(pool5, 256)
        deconv5 = self.unpool(deconv5, pool_indices5, tf.shape(conv13))

        deconv4 = self.conv_block(deconv5, 128)
        deconv4 = self.unpool(deconv4, pool_indices4, tf.shape(conv10))

        deconv3 = self.conv_block(deconv4, 64)
        deconv3 = self.unpool(deconv3, pool_indices3, tf.shape(conv7))

        deconv2 = self.conv_block(deconv3, 32)
        deconv2 = self.unpool(deconv2, pool_indices2, tf.shape(conv4))

        deconv1 = self.conv_block(deconv2, 16)
        deconv1 = self.unpool(deconv1, pool_indices1, tf.shape(conv2))

        # Output
        output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(deconv1)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

