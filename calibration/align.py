import cv2
import json
import numpy as np

from calibration.ideal_board import create_ideal_board


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


def align_with_reference_board(image):
    """
    Aligns the given image of an empty dartboard with an ideal dartboard.
    :param image: image of dartboard with no darts on it
    :return: homography matrix M
    """
    # resize image if necessary - parameters are currently tuned for 800x600
    h, w = image.shape
    if h != 600 or w != 800:
        image = cv2.resize(image, (800, 600))

    red_green_mask = extract_red_green_areas(image)

    contours, hier = cv2.findContours(red_green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    outer_ellipse_index = np.argmax(np.asarray([cv2.arcLength(c, False) for c in contours]))
    outer_ellipse_countour = contours[outer_ellipse_index]
    center, radii, rot = cv2.fitEllipse(outer_ellipse_countour)

    # increase radii by 5% for safe margin
    scaled_radii = tuple(i * 1.05 for i in radii)

    scoring_area_mask = cv2.ellipse(np.zeros_like(red_green_mask, dtype=np.uint8), (center, scaled_radii, rot), 255, -1)

    scoring_area = cv2.bitwise_and(image, image, mask=scoring_area_mask)
    gray_scoring_area = cv2.cvtColor(scoring_area, cv2.COLOR_BGR2GRAY)

    # find lines
    canny_edges = cv2.Canny(gray_scoring_area, 50, 200)
    lines = cv2.HoughLines(canny_edges, 1, np.pi / 360, 150)

    # find ten strongest pairwise different lines
    lines = _local_non_maximum_suppression(lines, num_maxima=10)

    # normalize angles
    for i in range(len(lines)):
        rho, theta = lines[i]
        if theta < 0:
            lines[i] = (-rho, theta + np.pi)

    # sort lines by angles
    lines = sorted(lines, key=lambda l: l[1])

    # heuristically adjust radii and center - or dont
    # radii = (radii[0] + 3, radii[1])
    # center = (center[0], center[1]-2)

    # compute intersections with outer ellipse
    outer_ellipse_intersections = np.zeros((20, 2))
    for i, (rho, theta) in enumerate(lines):
        r1, r2 = radii
        intersections = compute_ellipse_line_intersection((center, (r1 / 2, r2 / 2), rot), (rho, theta))
        if intersections is None:
            # TODO: handle this case
            raise Exception("Bad line!")
        p1, p2 = intersections
        outer_ellipse_intersections[i] = p1
        outer_ellipse_intersections[i + 10] = p2

    # for i, point in enumerate(outer_ellipse_intersections):
    #     x, y = point[:2].astype(np.int)
    #     cv2.circle(scoring_area, (x, y), 2, (0, 255, 0), -1)
    #     # cv2.putText(scoring_area, f'{i}', (x + 2, y + 2), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255))
    # cv2.imshow('scoring area', scoring_area)

    ideal_board = create_ideal_board()
    reference_points = ideal_board[5::6]
    # reference_points = 200 * (ideal_board[5::6] + 1)

    M, mask = cv2.findHomography(outer_ellipse_intersections, reference_points, cv2.RANSAC, 5.0)

    # transformed = cv2.warpPerspective(scoring_area, M, (800, 600))
    # cv2.circle(transformed, (200, 200), radius=200, color=(255, 255, 128), thickness=1)
    # cv2.circle(transformed, (200, 200), radius=200 * 107 // 170, color=(255, 255, 128), thickness=1)
    # cv2.circle(transformed, (200, 200), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.imshow('transformed', transformed)

    # cv2.waitKey()
    return M


def compute_ellipse_line_intersection(ellipse, line):
    """
    Computes intersections of given ellipse with line, if there are any
    Expects theta to be in [0, np.pi)
    :param ellipse: (center, radii, rotation)
    :param line: (rho, theta), i.e. Hough coordinates
    :return:
    """
    center, radii, rot = ellipse
    r1, r2 = radii
    rho, theta = line

    # center = (400, 300)
    # rot = 60
    # rho = 400
    # theta = np.pi/6
    # line = (rho, theta)

    # img = np.zeros((600, 800), dtype=np.uint8)
    # cv2.ellipse(img, (center, (2 * r1, 2*r2), rot), 255, 1)
    # cv2.circle(img, (int(center[0]), int(center[1])), radius=3, color=255)
    # a = np.cos(theta)
    # b = np.sin(theta)
    # x0 = a * rho
    # y0 = b * rho
    # pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    # pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    # cv2.line(img, pt1, pt2, 255, 1, cv2.LINE_AA)
    # cv2.imshow('ellipse and line before', img)

    # shift theta to [0, pi] by point reflection
    if theta < 0:
        theta = theta + np.pi
        rho = -rho

    # rotate line around ellipsis center such that the line and ellipse are parallel to axes (i.e. rot = 0)
    cx, cy = center
    rho = _distance_line_point((-rho, theta), (cx, cy))
    theta -= rot * np.pi / 180

    # from now on, work in coordinate system where ellipse is in center, i.e. don't execute the following line
    # rho = _distance_line_point((-rho, theta), (-cx, -cy))

    # if line is parallel to the y axis
    if np.isclose(theta, 0):
        x1, y1 = 0, r2
        x2, y2 = 0, -r2
    else:
        # use y = m * x + t representation
        m = np.tan(theta + np.pi / 2)
        t = rho / np.cos(np.pi / 2 - theta)

        # solve quadratic equation
        a = r1 ** 2 * m ** 2 + r2 ** 2
        b = 2 * r1 ** 2 * m * t
        c = r1 ** 2 * (t ** 2 - r2 ** 2)

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None

        x1 = (-b + np.sqrt(discriminant)) / (2 * a)
        y1 = m * x1 + t

        x2 = (-b - np.sqrt(discriminant)) / (2 * a)
        y2 = m * x2 + t

    # transform intersection points back to original coordinate system
    # x1, x2 = x1 + cx, x2 + cx
    # y1, y2 = y1 + cy, y2 + cy

    # rotate and then shift back
    alpha = rot * np.pi / 180 + np.pi
    R = np.asarray([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    center = np.asarray([cx, cy])
    p1 = R @ (np.asarray([x1, y1])) + center
    p2 = R @ (np.asarray([x2, y2])) + center

    # rho1, theta1 = line
    # img = np.zeros((600, 800), dtype=np.uint8)
    # cv2.ellipse(img, ((int(center[0]), int(center[1])), (2 * r1, 2*r2), rot), 255, 1)
    # cv2.circle(img, (int(center[0]), int(center[1])), radius=3, color=255)
    # a1, b1 = p1[:2]
    # a2, b2 = p2[:2]
    # cv2.circle(img, (int(a1), int(b1)), radius=5, color=255)
    # cv2.circle(img, (int(a2), int(b2)), radius=5, color=255)
    # a = np.cos(theta1)
    # b = np.sin(theta1)
    # x0 = a * rho1
    # y0 = b * rho1
    # pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    # pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    # cv2.line(img, pt1, pt2, 255, 1, cv2.LINE_AA)
    # cv2.imshow('ellipse and line after', img)
    #
    # rho = _distance_line_point((-rho, theta), (-cx, -cy))
    # img = np.zeros((600, 800), dtype=np.uint8)
    # cv2.ellipse(img, ((int(center[0]), int(center[1])), (2 * r1, 2*r2), 0), 255, 1)
    # cv2.circle(img, (int(center[0]), int(center[1])), radius=3, color=255)
    # x1, x2 = x1 + cx, x2 + cx
    # y1, y2 = y1 + cy, y2 + cy
    # cv2.circle(img, (int(x1), int(y1)), radius=5, color=255)
    # cv2.circle(img, (int(x2), int(y2)), radius=5, color=255)
    # a = np.cos(theta)
    # b = np.sin(theta)
    # x0 = a * rho
    # y0 = b * rho
    # pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    # pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    # cv2.line(img, pt1, pt2, 255, 1, cv2.LINE_AA)
    # cv2.imshow('ellipse and line unrotated after', img)

    # return point with rightmost x coordinate first
    if p2[0] > p1[0]:
        p1, p2 = p2, p1

    return p1, p2


def _distance_line_point(line, point):
    """
    Computes the smallest distance between the given line and point
    :param line: (rho, theta), i.e. Hough coordinates
    :param point: (x, y)
    :return: distance
    """
    rho, theta = line
    x, y = point
    # dist = | <w, x> + b |
    return np.abs(np.cos(theta) * x + np.sin(theta) * y + rho)


def _local_non_maximum_suppression(lines, num_maxima=10, rho_diff=10, angle_diff=3):
    """
    Determines the strongest lines from a list of HoughLines (i.e. list ordered by confidence).
    Suppresses local non-maxima by ensuring angle and position differences.
    :param lines: list of HoughLines
    :param num_maxima: number of lines to be returned
    :param rho_diff: minimum rho difference
    :param angle_diff: minimum angle difference
    :return: the strongest lines
    """
    if lines is None or len(lines) == 0:
        return
    nms_suppressed_lines = np.zeros((min(num_maxima, len(lines)), 2))
    nms_suppressed_lines[0] = np.asarray(lines[0])
    k = 0
    for i in range(min(num_maxima, len(lines) - 1)):
        while True:
            k += 1
            rho, theta = lines[k, 0]
            if rho < 0:
                rho *= -1
                theta -= np.pi
            closeness_rho = np.isclose(rho, nms_suppressed_lines[:, 0], atol=rho_diff)
            closeness_theta = np.isclose(theta, nms_suppressed_lines[:, 1], atol=2 * np.pi * angle_diff / 360)

            # if none is close, add line
            if np.all([~closeness_rho, ~closeness_theta]):
                nms_suppressed_lines[i] = np.array([rho, theta])
                break

    return nms_suppressed_lines


if __name__ == '__main__':
    image = cv2.imread('resources/cam0_2021_03_19_14_33_42_2908926.jpeg', cv2.IMREAD_COLOR)
    reference_image = cv2.imread('resources/pi0-distorted-sample.jpg', cv2.IMREAD_COLOR)

    image = cv2.resize(image, (800, 600))
    reference_image = cv2.resize(reference_image, (800, 600))

    image = white_balance(image)
    reference_image = white_balance(reference_image)

    cv2.imshow('orignal', image)
    # cv2.waitKey()
    align_with_reference_board(image)
    # align_with_reference_board(binary_image, reference_binary, None)
