from glob import glob
from os import path
import json
import cv2


def convert_labelme_json(labelme_json_dir, out_dir):
    """
    Extracts relevant information from labelme json files and stores them in more concise json files
    :param labelme_json_dir:
    :return:
    """
    labelme_json_dir = path.expanduser(labelme_json_dir)
    out_dir = path.expanduser(out_dir)
    for labelme_json_filename in glob(path.join(labelme_json_dir, '*.json')):
        with open(labelme_json_filename, 'rb') as labelme_json_file:
            labelme_json = json.load(labelme_json_file)
            data = {}
            data['imageName'] = path.basename(labelme_json['imagePath'])
            data['imageHeight'] = labelme_json['imageHeight']
            data['imageWidth'] = labelme_json['imageWidth']
            data['shapes'] = []
            for labelme_shape in labelme_json['shapes']:
                assert len(labelme_shape['points']) <=3
                shape = {
                    'label': labelme_shape['label'],
                    'point': labelme_shape['points'][0]}
                data['shapes'].append(shape)
        out_path = path.join(out_dir, '.'.join(data['imageName'].split('.')[:-1]) + '.json')
        if path.exists(out_path):
            print(
                f'File {path.basename(out_path)} already exists in directory {path.dirname(out_path)}!')
        else:
            with open(out_path, 'w', encoding='utf8') as out_file:
                json.dump(data, out_file, indent=2)


def add_empty_annotations(image_dir):
    image_dir = path.expanduser(image_dir)
    for image_filename in glob(path.join(image_dir, '*.jpeg')):
        image_name = path.basename(image_filename)
        json_path = path.join(image_dir, '.'.join(image_name.split('.')[:-1]) + '.json')
        if not path.exists(json_path):
            h, w = cv2.imread(image_filename, cv2.IMREAD_COLOR).shape[:2]
            data = {
                'imageName': path.basename(image_name),
                'imageHeight': h,
                'imageWidth': w,
                'shapes': []
            }
            with open(json_path, 'w', encoding='utf8') as json_file:
                json.dump(data, json_file, indent=2)
            print(f'Added empty annotation {path.basename(json_path)}')
