import cv2
import json
import numpy as np


def align_with_reference_board(image, intermediate_image, intermediate_keypoints):
    """
    :param intermediate_keypoints:
    :param image: undistorted image to be aligned
    :param intermediate_image: undistorted reference image
    :return:
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    intermediate_image = cv2.cvtColor(intermediate_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.SIFT_create()

    kp, desc = orb.detectAndCompute(image, None)
    ref_kp, ref_desc = orb.detectAndCompute(intermediate_image, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.match(desc, ref_desc)
    # its train image

    # homography = cv2.findHomography()
    final_img = cv2.drawMatches(image, kp,
                                intermediate_image, ref_kp, matches, None)
    final_img = cv2.resize(final_img, (2000, 1000))
    cv2.imshow('asdf', final_img)
    cv2.waitKey()


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


def compute_reference_transformation(intermediate_keypoints):
    """
    Aligns the
    :param intermediate_keypoints:
    :param ref_points:
    :return:
    """
    ref_width, ref_height, ref_points = load_reference_data()
    transformation_matrix = cv2.getPerspectiveTransform(intermediate_keypoints, ref_points)
    return transformation_matrix


if __name__ == '__main__':
    image = cv2.imread('resources/board1.jpg', cv2.IMREAD_COLOR)
    intermediate_image = cv2.imread('resources/pi0-distorted-sample.jpg', cv2.IMREAD_COLOR)

    align_with_reference_board(image, intermediate_image, None)
