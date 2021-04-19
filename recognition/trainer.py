import tensorflow as tf
from tqdm import tqdm

from recognition.evaluation import evaluate_multi_metric
from recognition.losses import hungarian_loss
from recognition.unet_hires.model import UNetHiRes


class Trainer(object):
    def __init__(self, log_dir, train_dataset, eval_dataset):
        self.epochs = 1000
        self.model = UNetHiRes()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.mu = 1
        self.rho = 1
        self.loss = hungarian_loss
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.log_dir = log_dir

        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def train_step(self, images, ground_truth):
        """
        Performs one training step on the given batch of images and ground truths
        :param images:
        :param ground_truth:
        :return:
        """
        predictions = self.model(images, training=True)
        loss = self.loss(predictions, ground_truth, mu=self.mu, rho=self.rho)
        self.optimizer.minimize(loss, self.model.trainable_variables)
        return loss

    def train(self):
        train_iterator = iter(self.train_dataset)
        for epoch in range(self.epochs):
            losses = []
            for _ in range(10):
                images, ground_truths = next(train_iterator)
                loss = self.train_step(images, ground_truths)
                losses.append(loss)
            avg_training_loss = tf.reduce_mean(losses)
            tf.summary.scalar('loss_train/total', avg_training_loss,
                              step=tf.compat.v1.train.get_or_create_global_step())

            detect_loss, distance_loss, field_loss = self.evaluate()
            tf.summary.scalar('loss_eval/detect', detect_loss, step=tf.compat.v1.train.get_or_create_global_step())
            tf.summary.scalar('loss_eval/distance', distance_loss, step=tf.compat.v1.train.get_or_create_global_step())
            tf.summary.scalar('loss_eval/field', field_loss, step=tf.compat.v1.train.get_or_create_global_step())

            print()
            print(f'Epoch {epoch}, '
                  f'Training Loss: {avg_training_loss}'
                  f'Evaluation Loss: Detect {detect_loss}'
                  f'Evaluation Loss: Distance {distance_loss}'
                  f'Evaluation Loss: Field {field_loss}')

    def evaluate(self):
        predictions = []
        annotations = []
        for image, annotation in self.eval_dataset:
            predictions.extend(self.model(image, training=False))
            annotations.extend(annotation)
        return evaluate_multi_metric(predictions, annotations)



