from os import path
import json
from glob import glob


def sanity_check(json_dir):
    """
    Check json files for sanity.
    :param json_dir:
    :return:
    """
    json_dir = path.expanduser(json_dir)
    for json_filename in glob(path.join(json_dir, '*.json')):
        with open(json_filename, 'r') as json_file:
            annotation = json.load(json_file)
            # Check existence of keys
            for key in {'imageName', 'imageHeight', 'imageWidth', 'shapes'}:
                if key not in annotation:
                    print(f'{key} is missing in {json_filename}.')

            # Check shapes
            shapes = annotation['shapes']
            if not isinstance(shapes, list):
                print(f'Wrong shapes type in {json_filename}.')

            for shape in shapes:
                for key in {'label', 'point', 'score'}:
                    if key not in shape:
                        print(f'{key} is missing in shapes in {json_filename}.')

            # Check existence of image file
            if not path.exists(path.join(json_dir, annotation['imageName'])):
                print(f"Image {annotation['imageName']} does not exist in {json_filename}!")

    print('Sanity check completed.')
