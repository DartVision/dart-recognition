import cv2
import json
import numpy as np


def extract_red_green_areas(image):
    """

    :param grayscale_image:
    :return:
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_OTSU)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_lower_threshold1 = np.array([0, 120, 100])
    red_upper_threshold1 = np.array([5, 255, 255])

    red_lower_threshold2 = np.array([160, 120, 100])
    red_upper_threshold2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv_image, red_lower_threshold1, red_upper_threshold1)
    red_mask2 = cv2.inRange(hsv_image, red_lower_threshold2, red_upper_threshold2)
    red_binary = red_mask1 | red_mask2

    green_lower_threshold = np.array([60, 110, 60])
    green_upper_threshold = np.array([75, 255, 255])

    green_binary = cv2.inRange(hsv_image, green_lower_threshold, green_upper_threshold)
    result = red_binary | green_binary

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result


def white_balance(img):
    """
    White balances using the gray-world assumption
    :param img:
    :return:
    """
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def align_with_reference_board(image, intermediate_image, intermediate_keypoints):
    """
    :param intermediate_keypoints:
    :param image: undistorted image to be aligned
    :param intermediate_image: undistorted reference image
    :return:
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # intermediate_image = cv2.cvtColor(intermediate_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()

    kp, desc = orb.detectAndCompute(image, None)
    ref_kp, ref_desc = orb.detectAndCompute(intermediate_image, None)

    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = matcher.match(desc, ref_desc)
    matches = sorted(matches, key=lambda x: x.distance)[:20]
    # homography = cv2.findHomography()
    final_img = cv2.drawMatches(image, kp,
                                intermediate_image, ref_kp, matches, None)
    final_img = cv2.resize(final_img, (2000, 1000))
    cv2.imshow('asdf', final_img)
    cv2.waitKey()


def load_reference_data():
    with open('../resources/pi0-distorted-sample.json') as json_file:
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


def align_binary_with_reference(red_green_mask, binary_reference_image, color_image):
    corner_algo = lambda image: cv2.goodFeaturesToTrack(image, 25, 0.01, 10)

    canny_edges = cv2.Canny(red_green_mask, 50, 200)

    cv2.imshow('canny', canny_edges)

    contours, hier = cv2.findContours(red_green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(color_image, contours, -1, (0, 255, 0), 1)
    # cv2.imshow('contours', color_image)
    # cv2.waitKey()

    outer_ellipse_index = np.argmax(np.asarray([cv2.arcLength(c, False) for c in contours]))
    outer_ellipse_countour = contours[outer_ellipse_index]
    center, radii, rot = cv2.fitEllipse(outer_ellipse_countour)

    # increase radii by 5% for safe margin
    radii = tuple(i * 1.05 for i in radii)

    scoring_area_mask = cv2.ellipse(np.zeros_like(red_green_mask, dtype=np.uint8), (center, radii, rot), 255, -1)

    # color_image = cv2.cvtColor(red_green_mask.copy(), cv2.COLOR_GRAY2BGR)

    cv2.ellipse(color_image, (center, radii, rot), (0, 0, 255), 1)

    cv2.imshow('Outer ellipse', scoring_area_mask)
    # corners = corner_algo(binary_image)

    # best_corners = np.argwhere(corners > 0.2 * np.max(corners))

    # reference_corners = corner_algo(binary_reference_image)
    # best_reference_corners = np.argwhere(reference_corners > 0.2 * np.max(reference_corners))

    # M, mask = cv2.findHomography(best_corners[:50], best_reference_corners[:50], cv2.RANSAC, 5.0)

    # final_img = cv2.drawMatches(image, kp,
    #                             intermediate_image, ref_kp, matches, None)
    # final_img = cv2.resize(final_img, (2000, 1000))

    # corners = cv2.dilate(corners, None)
    # colored = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    # colored[corners > 0.2 * np.max(corners)] = [0, 0, 255]
    # corners = colored
    # cv2.imshow('corners', corners)
    cv2.waitKey()


if __name__ == '__main__':
    image = cv2.imread('resources/cam0_2021_03_19_14_33_42_2908926.jpeg', cv2.IMREAD_COLOR)
    reference_image = cv2.imread('resources/pi0-distorted-sample.jpg', cv2.IMREAD_COLOR)

    image = cv2.resize(image, (800, 600))
    reference_image = cv2.resize(reference_image, (800, 600))

    image = white_balance(image)
    reference_image = white_balance(reference_image)

    cv2.imshow('orignal', image)
    # cv2.waitKey()
    binary_image = extract_red_green_areas(image)
    reference_binary = extract_red_green_areas(reference_image)
    align_binary_with_reference(binary_image, reference_binary, image)
    # align_with_reference_board(binary_image, reference_binary, None)
