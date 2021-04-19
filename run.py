import tensorflow as tf

from recognition.load_dataset import prepare_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = prepare_dataset(augment=False)