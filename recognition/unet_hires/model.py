import tensorflow as tf

IMAGE_SIZE = 300


def mobile_net_v2_encoder():
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3],
                                                   weights='imagenet',
                                                   include_top=False)
    layer_names = ["block_1_expand_relu",
                   "block_3_expand_relu",
                   "block_6_expand_relu",
                   "block_13_expand_relu",
                   "block_16_project"
                   ]
    output_layers = [mobile_net.get_layer(name).output for name in layer_names]
    encoder = tf.keras.Model(inputs=mobile_net.input, outputs=output_layers)
    encoder.trainable = False
    return encoder


class UNetHiRes(tf.keras.models.Model):
    def __init__(self):
        super(UNetHiRes, self).__init__()
        self.encoder = mobile_net_v2_encoder()

        self.transpose_conv_1 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3,
                                                                strides=2, padding='same', use_bias=False)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

        self.transpose_conv_2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3,
                                                                strides=2, padding='same', use_bias=False)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        self.transpose_conv_3 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3,
                                                                strides=2, padding='same', use_bias=False)
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.transpose_conv_4 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3,
                                                                strides=2, padding='same', use_bias=False)
        self.batch_norm_4 = tf.keras.layers.BatchNormalization()

        # Final dense layer
        # First two units are for classification, third and fourth for location, last 4 for background color of dart
        # 3 predictions maximum
        self.final_layer = tf.keras.layers.Dense(3*8)

    def call(self, inputs, training=None, **kwargs):
        # Get output from all five output layers of the encoder
        encoder_outputs = self.encoder(inputs, training=training)

        x = encoder_outputs[-1]
        x = self.transpose_conv_1(x, training=training)
        x = self.batch_norm_1(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, encoder_outputs[-2]], axis=-1)

        x = self.transpose_conv_2(x, training=training)
        x = self.batch_norm_2(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, encoder_outputs[-3]], axis=-1)

        x = self.transpose_conv_3(x, training=training)
        x = self.batch_norm_3(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, encoder_outputs[-4]], axis=-1)

        x = self.transpose_conv_4(x, training=training)
        x = self.batch_norm_4(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, encoder_outputs[-5]], axis=-1)

        x = self.final_layer(x, training=training)

        return tf.reshape(x, (3, 8))
