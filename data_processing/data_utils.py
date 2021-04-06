from shutil import copyfile
from glob import glob
from os import path
import json


def copy_images_and_annotations(src_dir, out_dir, prefix=''):
    """
    Copies images and annotations from src_dir to out_dir, prepending the given prefix.
    Also changes imageName in annotation files to new image name
    :param src_dir:
    :param out_dir:
    :param prefix:
    :return:
    """
    copy_annotations(src_dir, out_dir, prefix)
    copy_images(src_dir, out_dir, prefix)


def copy_annotations(src_dir, out_dir, prefix=''):
    """
    Copies annotations from src_dir to out_dir, prepending the given prefix.
    Changes imageName in annotation files by also prepending the given prefix.
    :param src_dir:
    :param out_dir:
    :param prefix:
    :return:
    """
    src_dir = path.expanduser(src_dir)
    out_dir = path.expanduser(out_dir)
    for json_path in glob(path.join(src_dir, '*.json')):
        with open(json_path, 'rb') as json_file:
            annotation = json.load(json_file)
        filename = path.basename(json_path)
        if prefix and prefix != '':
            annotation['imageName'] = f"{prefix}_{annotation['imageName']}"
            filename = f"{prefix}_{filename}"
        out_file_path = path.join(out_dir, filename)
        if path.exists(out_file_path):
            print(
                f'File {path.basename(out_file_path)} already exists in directory {path.dirname(out_file_path)}!')
        else:
            with open(out_file_path, 'w', encoding='utf8') as out_file:
                json.dump(annotation, out_file)


def copy_images(src_dir, out_dir, prefix=''):
    """
    Copies images from src_dir to out_dir, prepending the given prefix.
    :param src_dir:
    :param out_dir:
    :param prefix:
    :return:
    """
    src_dir = path.expanduser(src_dir)
    out_dir = path.expanduser(out_dir)
    image_files = [f for f_ in [glob(path.join(src_dir, e)) for e in ('*.jpeg', '*.jpg', '*.png')] for f in f_]
    for image_path in image_files:
        out_filename = path.basename(image_path)
        if prefix and prefix != '':
            out_filename = f"{prefix}_{out_filename}"

        out_file_path = path.join(out_dir, out_filename)
        if path.exists(out_file_path):
            print(
                f'File {path.basename(out_file_path)} already exists in directory {path.dirname(out_file_path)}!')
        else:
            copyfile(image_path, out_file_path)


if __name__ == '__main__':
    copy_images_and_annotations('~/datasets/darts/raw/pi1', '~/datasets/darts/data', 'cam1')
