from glob import glob
from os import path
import json
import cv2
import numpy as np
from enum import Enum


def start_annotate_score(images_dir, json_dir, out_dir):
    """
    Starts workflow for annotating the score.
    Assumes that json_dir contains json files that are structured as given by `convert_labelme.py`.
    Images are searched in directory `images_dir`.
    If json_out_dir is the same as json_dir, the json files are overwritten.

    The input jsons should look like:
    {
      "imageName": "asdf.jpg",
      "imageHeight": 123,
      "imageWidth": 123,
      "shapes": [ { "label": "xyz", "point": [ 123, 123 ] } ]
    }

    This workflow creates json files that look exactly the same, except each shape in "shapes" gets a property
    "score"; a string with a value in 0,1,...,20,25,D1,D2,...,D20,D25,T1,T2,...,T20.

    Manual for annotation process:
    A dart with an outlined circle is the currently selected dart for annotating score;
    a dart with a filled circle represents an unselected dart.
    You can enter a score for the selected dart using keys d,t,0,...,9.
    You can confirm a score using space. The entered score will only be accepted, if it is a valid option
    as declared above.
    If you want to skip a dart, press k.
    If you want to quit the whole process, press q or simply close the window.
    :return:
    """
    window = 'window'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # annotate(window, img_path, [], 1)
    for json_filename in glob(path.join(json_dir, '*.json')):
        with open(json_filename, 'rb') as json_file:
            old_json = json.load(json_file)
        img_path = path.join(images_dir, old_json["imageName"])
        for i in range(len(old_json["shapes"])):
            (res, score) = annotate(window, img_path, old_json["shapes"], i)
            if res == AnnotationResult.QUIT:
                cv2.destroyAllWindows()
                return
            elif res == AnnotationResult.SKIP:
                continue
            elif res == AnnotationResult.SUCCESSFUL:
                old_json["shapes"][i]["score"] = score
                out_path = path.join(out_dir, json_filename)
                with open(out_path, 'w', encoding='utf8') as out_file:
                    json.dump(old_json, out_file, indent=2)

    cv2.destroyAllWindows()


class AnnotationResult(Enum):
    SUCCESSFUL = 1
    SKIP = 2
    QUIT = 3


def is_valid_score(score):
    if len(score) == 0:
        return False
    if score[0] == 'D':
        if not score[1:].isdigit():
            return False
        return int(score[1:]) in range(1, 21) or int(score[1:]) == 25
    if score[0] == 'T':
        if not score[1:].isdigit():
            return False
        return int(score[1:]) in range(1, 21)
    if score.isdigit():
        return int(score) in range(0, 21) or int(score) == 25
    return False


def annotate(window, img_path, shapes, annotate_index):
    picture = cv2.imread(img_path, cv2.IMREAD_COLOR)
    score = shapes[annotate_index]["score"] if "score" in shapes[annotate_index].keys() else ''

    def draw():
        image = np.copy(picture)
        color = (255, 50, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(shapes)):
            point = (round(shapes[i]["point"][0]), round(shapes[i]["point"][1]))
            cv2.circle(image, point, 32 if i == annotate_index else 8, color, 8 if i == annotate_index else -1)
            if i == annotate_index:
                cv2.putText(image, score, (point[0] + 48, point[1] + 20),
                            font, 2, (0, 0, 0), 16, cv2.LINE_AA)
                cv2.putText(image, score, (point[0] + 48, point[1] + 20),
                            font, 2, color, 8, cv2.LINE_AA)
            else:
                json_score = shapes[i]["score"] if "score" in shapes[i].keys() else ''
                cv2.putText(image, json_score, (point[0] + 16, point[1] + 10),
                            font, 1, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(image, json_score, (point[0] + 16, point[1] + 10),
                            font, 1, color, 2, cv2.LINE_AA)
        cv2.imshow(window, image)

    draw()
    while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
        try:
            k = cv2.waitKey(50)
            key = chr(k)
        except ValueError:
            continue

        if key == 'q':
            return AnnotationResult.QUIT, score
        elif key == 'k':
            return AnnotationResult.SKIP, score
        elif key == ' ':
            if is_valid_score(score):
                return AnnotationResult.SUCCESSFUL, score
            else:
                print("Annotation invalid. If you want to skip, press k.")
        elif key.isdigit() or key in ["d", "t"]:
            score = (score + key).upper()
            draw()
        elif key == '\b':
            score = score[:-1]
            draw()
    return AnnotationResult.QUIT, score
