import tensorflow as tf
from os import path, makedirs

from recognition.evaluation import evaluate_multi_metric
from recognition.losses import hungarian_loss
from recognition.models.mobile_net_v2 import MobileNetV2MacroDetector
from recognition.models.unet_hires import UNetHiRes


class Trainer(object):
    """
    Class that allows training the model and logging the loss and error metrics
    """

    def __init__(self, log_dir, checkpoint_dir, experiment_name, train_dataset, eval_dataset, image_size):
        self.epochs = 1001
        self.model = MobileNetV2MacroDetector(image_size=image_size)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
        self.mu = 1
        self.rho = 1
        self.loss = hungarian_loss
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.log_dir = path.join(path.expanduser(log_dir), experiment_name)
        self.checkpoint_dir = path.join(path.expanduser(checkpoint_dir), experiment_name)
        makedirs(self.log_dir, exist_ok=True)
        makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpoint_interval = 40

        self.global_step = tf.Variable(0, name='global_step', dtype=tf.int32)

        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.summary_writer.set_as_default()

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              global_step=self.global_step,
                                              model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir,
                                                             checkpoint_interval=self.checkpoint_interval,
                                                             step_counter=self.global_step,
                                                             max_to_keep=None)

    def train_step(self, images, ground_truth):
        """
        Performs one training step on the given batch of images and ground truths
        :param images:
        :param ground_truth:
        :return:
        """
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss(predictions, ground_truth, mu=self.mu, rho=self.rho, global_step=self.global_step)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        tf.summary.scalar('train/total', loss, step=self.global_step.numpy())
        self.checkpoint_manager.save(check_interval=True)

        # increment global step
        self.global_step.assign(self.global_step.value() + 1)
        return loss

    def train(self):
        train_iterator = iter(self.train_dataset)
        for epoch in range(self.epochs):

            images, ground_truths = next(train_iterator)
            training_loss = self.train_step(images, ground_truths)
            print()
            print(f'Epoch {epoch}, '
                  f'Training Loss: {training_loss}')

            if self.global_step.value() != 0 and self.global_step.value() % 5 == 0:
                detect_loss, distance_loss, field_loss = self.evaluate()
                tf.summary.scalar('eval/detection', detect_loss, step=self.global_step.numpy())
                tf.summary.scalar('eval/location', distance_loss, step=self.global_step.numpy())
                tf.summary.scalar('eval/field', field_loss, step=self.global_step.numpy())
                print(f'Evaluation Loss: Detection {detect_loss}\n'
                      f'Evaluation Loss: Location {distance_loss}\n'
                      f'Evaluation Loss: Field {field_loss}')

    def evaluate(self):
        predictions = []
        annotations = []
        for image, annotation in self.eval_dataset:
            predictions.extend(self.model(image, training=False))
            annotations.extend(annotation)
        predictions = tf.convert_to_tensor(predictions)
        annotations = tf.convert_to_tensor(annotations)
        return evaluate_multi_metric(predictions, annotations)
