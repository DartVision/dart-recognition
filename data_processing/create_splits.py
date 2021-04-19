from os import path
from glob import glob
import random


def create_train_test_split(data_dir, test_percentage=0.25):
    """
    Splits the data into train and test images. Creates two files in the data directory, test.txt and train.txt,
    containing the image names of the test and train images, respectively
    :param data_dir:
    :param test_percentage:
    :return:
    """
    data_dir = path.expanduser(data_dir)
    annotations = glob(path.join(data_dir, '*.json'))
    num_test_images = int(test_percentage * len(annotations))

    random.shuffle(annotations)
    test_annotations = annotations[:num_test_images]
    train_annotations = annotations[num_test_images:]

    with open(path.join(data_dir, 'test.txt'), 'w') as file:
        file.write('\n'.join(test_annotations))
    with open(path.join(data_dir, 'train.txt'), 'w') as file:
        file.writelines('\n'.join(train_annotations))

    print(f'Created train and test split with {len(train_annotations)} train '
          f'images and {len(test_annotations)} test images.')
