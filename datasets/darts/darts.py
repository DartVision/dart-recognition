"""darts dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
from os import path
import json
from enum import Enum

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

PATH = '~/datasets/darts/data'


class COLORS(Enum):
    BLACK = 0
    WHITE = 1
    RED = 2
    GREEN = 3


SCORE_COLORS = {}
_color_scores = [{str(i): COLORS.BLACK for i in [20, 18, 13, 10, 2, 3, 7, 8, 14, 12]},
                 {str(i): COLORS.WHITE for i in [1, 4, 6, 15, 17, 19, 16, 11, 9, 5]},
                 {str(i): COLORS.RED for i in
                  ['D1', 'T1', 'D4', 'T4', 'D6', 'T6', 'D15', 'T15', 'D17', 'T17', 'D19', 'T19',
                   'D11', 'T11', 'D9', 'T9', 'D5', 'T5', '25']},
                 {str(i): COLORS.BLACK for i in
                  ['D20', 'T20', 'D18', 'T18', 'D13', 'T13', 'D10', 'T10', 'D2', 'T2',
                   'D7', 'T7', 'D14', 'T14', 'D12', 'T12', '50']}]

for color_scores in _color_scores:
    SCORE_COLORS.update(color_scores)


class Darts(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for darts dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(None, None, 3)),
                'labels': tfds.features.Sequence(
                    {
                        'x': tf.float32,
                        'y': tf.float32,
                        'color': tfds.features.ClassLabel(num_classes=4)
                    }),
            }),
            supervised_keys=('image', 'labels'),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        data_path = path.expanduser(PATH)
        return {
            'train': self._generate_examples(data_path, 'train'),
            'test': self._generate_examples(data_path, 'test')
        }

    def _generate_examples(self, directory, split):
        """Yields examples."""
        with open(path.join(directory, f'{split}.txt'), 'r') as split_file:
            split_names = split_file.readlines()
        for f in split_names:
            json_name = path.join(directory, f)
            with open(json_name, 'rb') as json_file:
                annotation = json.load(json_file)
                image_name = annotation['imageName']
                w, h = annotation['imageWidth'], annotation['imageHeight']
                labels = []
                for shape in annotation['shapes']:
                    if shape['label'] == 'dart':
                        x, y = shape['point']
                        label = {
                            'x': x / w,
                            'y': y / h,
                            'color': SCORE_COLORS[shape['score']]
                        }
                        labels.append(label)
                yield f, {
                    'image': image_name,
                    'label': labels,
                }
