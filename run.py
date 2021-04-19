import tensorflow as tf

from recognition.load_dataset import prepare_dataset
from recognition.trainer import Trainer

if __name__ == '__main__':
    image_size = 224
    train_dataset, test_dataset = prepare_dataset(augment=False, size=image_size)

    checkpoint_dir = '/home/nikolas/Projects/dart-recognition/training/checkpoints'
    log_dir = '/home/nikolas/Projects/dart-recognition/training/logs'

    experiment_name = 'first_test'

    trainer = Trainer(log_dir=log_dir, checkpoint_dir=checkpoint_dir, train_dataset=train_dataset,
                      eval_dataset=test_dataset, image_size=image_size, experiment_name=experiment_name)
    trainer.train()
