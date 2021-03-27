import tensorflow as tf
from tqdm import tqdm

from recognition.unet_hires.losses import hungarian_loss
from recognition.unet_hires.model import UNetHiRes


class Trainer(object):
    def __init__(self):
        self.epochs = 1000
        self.model = UNetHiRes()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.mu = 1
        self.loss = hungarian_loss
        self.train_dataset = None
        self.eval_dataset = None

    def train_step(self, images, ground_truth):
        """
        Performs one training step on the given batch of images and ground truths
        :param images:
        :param ground_truth:
        :return:
        """
        predictions = self.model(images, training=True)
        loss = self.loss(images, ground_truth, mu=self.mu)
        self.optimizer.minimize(loss, self.model.trainable_variables)
        return loss

    def train(self):
        for epoch in range(self.epochs):
            losses = []
            for step, (images, ground_truths) in enumerate(tqdm(self.train_dataset)):
                loss = self.train_step(images, ground_truths)
                losses.append(loss.numpy())
