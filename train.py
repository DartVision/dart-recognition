import tensorflow as tf

from recognition.load_dataset import prepare_dataset
from recognition.train.macro_trainer import Trainer

if __name__ == '__main__':

    # limit gpu memory allocation
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(6)

    image_size = 224
    train_dataset, test_dataset = prepare_dataset(augment=True, size=image_size)

    checkpoint_dir = '/home/nikolas/Projects/dart-recognition/training/checkpoints'
    log_dir = '/home/nikolas/Projects/dart-recognition/training/logs'

    experiment_name = 'mobile_net_avg_pooling_augment'

    trainer = Trainer(log_dir=log_dir, checkpoint_dir=checkpoint_dir, train_dataset=train_dataset,
                      eval_dataset=test_dataset, image_size=image_size, experiment_name=experiment_name)
    trainer.train()
