from os import path
from glob import glob
import tensorflow as tf
import tensorflow_datasets as tfds
import imgaug.augmenters as iaa
import imgaug.augmentables as iag
import numpy as np


def prepare_dataset(batch_size=64, augment=False, size=500):
    test_dataset, train_dataset = tfds.load('darts', split=['test', 'train'])

    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.map(extract_image_and_annotation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(lambda i, a: resize_image(i, a, size),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if augment:
        train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(batch_size) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = test_dataset.map(extract_image_and_annotation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(lambda i, a: resize_image(i, a, size),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.shuffle(buffer_size=100).batch(batch_size) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset


def extract_image_and_annotation(datapoint):
    image, labels = datapoint['image'], datapoint['labels']
    annotation = tf.stack(
        [tf.ones(tf.shape(labels['x'])[0]), labels['x'], labels['y'], tf.cast(labels['color'], dtype=tf.float32)],
        axis=1)
    # add dummy elements if necessary
    num_annots = tf.shape(labels['x'])[0]
    if num_annots < 3:
        dummy = tf.convert_to_tensor([[0, 0, 0, 0]], tf.float32)
        dummy_annotations = tf.repeat(dummy, repeats=[3 - num_annots], axis=0)
        annotation = tf.concat([annotation, dummy_annotations], axis=0)

    return image, annotation


def annotation_to_keypoints(image, annotation):
    h, w = image.shape[:2]
    keypoints = iag.KeypointsOnImage(
        [iag.Keypoint(round(w / 2 * (label[1] + 1)), round(h / 2 * (label[2] + 1))) for label in annotation],
        shape=image.shape[:2])
    return keypoints


def keypoints_to_annotation(image, annotation, keypoints):
    for i, kp in enumerate(keypoints.to_xy_array()):
        dims = image.shape[:2][::-1]
        if annotation[i, 0] > 0.5:
            annotation[i, 1:3] = (2 * kp - dims) / dims
    return annotation


def augment_image(image, annotation):
    def np_func(img, ann):
        # convert to imgaug keypoints
        keypoints = annotation_to_keypoints(img, ann)
        img = img.astype(np.uint8)

        p = 0.1
        seq = iaa.Sequential([
            iaa.Sometimes(p, iaa.Sequential([iaa.ShearY((-20, 20))])),
            iaa.Sometimes(p, iaa.ChangeColorTemperature((3500, 8000))),
            iaa.Sometimes(p, iaa.AddToBrightness((-15, 15))),
            iaa.Sometimes(p, iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255), per_channel=True))
        ])

        img, keypoints = seq(image=img, keypoints=keypoints)

        # convert from imgaug keypoints
        ann = keypoints_to_annotation(img, ann, keypoints)

        return img.astype(np.float32), ann

    image_shape, annotation_shape = tf.shape(image), tf.shape(annotation)
    image, annotation = tf.numpy_function(np_func, [image, annotation], Tout=[image.dtype, annotation.dtype])
    image, annotation = tf.reshape(image, image_shape), tf.reshape(annotation, annotation_shape)
    return image, annotation


def resize_image(image, annotation, size):
    image = tf.image.resize(image, (size, size), method=tf.image.ResizeMethod.BILINEAR)
    return image, annotation


def normalize_image(image, annotation):
    image = tf.cast(image, tf.float32) / 255
    return image, annotation


if __name__ == '__main__':
    train_dataset, test = prepare_dataset(augment=True)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import numpy as np

    train_iterator = iter(train_dataset)
    num_vis = 3
    plt.figure(figsize=(7, 7))
    for i in range(num_vis * 2):
        image, annotation = next(train_iterator)
        if i == 0:
            print("Image shape: ", image.shape)
        # Visualize first example in each batch
        image = image[0].numpy()
        annotation = annotation[0].numpy()
        ax = plt.subplot(num_vis, 2, i + 1)
        ax.imshow(image)
        for a in annotation:
            if a[0] > 0.5:
                x, y = a[1:3]
                h, w = image.shape[:2]
                x = np.round(w / 2 * (x + 1))
                y = np.round(h / 2 * (y + 1))
                circ = Circle((x, y), 3)
                ax.add_patch(circ)

    plt.show()
    del (train_iterator)
