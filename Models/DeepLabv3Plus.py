import tensorflow as tf

class DeepLabv3Plus:
    def __init__(self, inputs):
        self.Model = self.build_model(inputs, 1)

    def conv_block(self, inputs, filters, kernel_size=3, strides=1, padding='same', activation='relu', dilation_rate=1):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        return x

    def atrous_spatial_pyramid_pooling(self, inputs, filters):
        # ASPP with parallel dilated convolutions
        conv1 = self.conv_block(inputs, filters, kernel_size=1)

        conv2 = self.conv_block(inputs, filters, kernel_size=3, dilation_rate=6)

        conv3 = self.conv_block(inputs, filters, kernel_size=3, dilation_rate=12)

        conv4 = self.conv_block(inputs, filters, kernel_size=3, dilation_rate=18)

        # Global average pooling
        pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        pool = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(pool)
        pool = self.conv_block(pool, filters, kernel_size=1)

        # Dynamically calculate the upsample size
        upsample_size = tf.keras.backend.int_shape(inputs)[1:3]
        pool = tf.keras.layers.UpSampling2D(size=upsample_size, interpolation='bilinear')(pool)

        # Concatenate and apply 1x1 convolution
        concat = tf.keras.layers.concatenate([conv1, conv2, conv3, conv4, pool])
        output = self.conv_block(concat, filters, kernel_size=1)

        return output

    def build_model(self, inputs, num_classes):
        # Encoder
        backbone = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
        # backbone.summary()
        encoder_output = backbone.get_layer('conv4_block6_out').output

        # Atrous Spatial Pyramid Pooling
        aspp_output = self.atrous_spatial_pyramid_pooling(encoder_output, filters=256)

        # Upsampling and skip connections
        upsample1 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(aspp_output)
        upsample1 = self.conv_block(upsample1, filters=256, kernel_size=1)

        skip_connection = backbone.get_layer('conv2_block3_out').output
        skip_connection = self.conv_block(skip_connection, filters=48, kernel_size=1)

        # # Upsample skip_connection to match the shape of upsample1
        # skip_connection = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(skip_connection)

        merge1 = tf.keras.layers.concatenate([upsample1, skip_connection])
        merge1 = self.conv_block(merge1, filters=256, kernel_size=3)

        # upsample2 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(merge1)
        # upsample2 = self.conv_block(upsample2, filters=128, kernel_size=1)

        # skip_connection = backbone.get_layer('conv2_block3_out').output
        # skip_connection = self.conv_block(skip_connection, filters=24, kernel_size=1)
        #
        # merge2 = tf.keras.layers.concatenate([upsample2, skip_connection])
        # merge2 = self.conv_block(merge2, filters=128, kernel_size=3)

        # Final convolution and upsampling
        output = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(merge1)
        output = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(output)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model