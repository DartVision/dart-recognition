import tensorflow as tf


def mobile_net_v2_encoder(image_size):
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=[image_size, image_size, 3],
                                                   weights='imagenet',
                                                   include_top=False,
                                                   pooling='avg')

    encoder = tf.keras.Model(inputs=mobile_net.input, outputs=mobile_net.outputs)
    encoder.trainable = False
    return encoder


class MobileNetV2Detector(tf.keras.models.Model):
    def __init__(self, image_size):
        super(MobileNetV2Detector, self).__init__()
        self.encoder = mobile_net_v2_encoder(image_size=image_size)

        # Final dense layer
        # First two units are for classification, third and fourth for location, last 4 for background color of dart
        # 3 predictions maximum
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(1024, use_bias=False)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.final_layer = tf.keras.layers.Dense(3 * 8, use_bias=True)

    def call(self, inputs, training=None, **kwargs):
        x = self.encoder(inputs, training=training)
        x = self.flatten(x)
        x = self.dense_1(x, training=training)
        x = self.batch_norm_1(x, training=training)
        x = tf.nn.relu(x)
        x = self.final_layer(x, training=training)

        return tf.reshape(x, (-1, 3, 8))
