import cv2
import numpy as np
import json


def load_reference_data():
    with open('../resources/reference-board.json') as json_file:
        ref_board = json.load(json_file)
        width = ref_board["width"]
        height = ref_board["height"]
        points = np.array([
            ref_board["points"]["20L"],
            ref_board["points"]["6T"],
            ref_board["points"]["3R"],
            ref_board["points"]["8T"]
        ], np.float32)
        return width, height, points


ref_width, ref_height, ref_points = load_reference_data()


def transform_image(image, point20l, point6t, point3r, point8t):
    mtx = cv2.getPerspectiveTransform(
        np.array([point20l, point6t, point3r, point8t], np.float32),
        ref_points
    )
    return cv2.warpPerspective(image, mtx, (ref_width, ref_height))


if __name__ == '__main__':
    sample = cv2.imread("../resources/pi0-distorted-sample.jpg")
    with open('../resources/pi0-distorted-sample.json') as json_file:
        sample_board = json.load(json_file)
        transformed = transform_image(sample, sample_board["points"]["20L"], sample_board["points"]["6T"],
                                      sample_board["points"]["3R"], sample_board["points"]["8T"])
        cv2.imwrite("distorted-and-transformed.jpg", transformed)

