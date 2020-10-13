import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from matplotlib import colors
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.ndimage import label


def extract_overlaps(mask1, mask2):
    """
    Extracts the connected components of mask2 that overlap with mask1
    :param mask1:
    :param mask2:
    :return:
    """
    mask2 = np.copy(mask2)
    labeled_image, num_labels = label(mask2 > 127)

    for i in range(1, num_labels + 1):
        if np.count_nonzero(cv2.bitwise_and(mask1, mask2, mask=np.asarray(labeled_image == i, dtype=np.uint8))) == 0:
            mask2[labeled_image == i] = 0
    return mask2


def reduce_color_depth(image, n):
    """
    Reduces color depth to n values per channel
    :param image:
    :param n:
    :return:
    """
    img = image.astype(np.float32)
    return (np.around(img / 255 * n - 0.5) / n * 255 + 255 / (2 * n)).astype(np.uint8)


def extract_mask(image, color):
    return (image[:, :, 0] == color[0]) & (image[:, :, 1] == color[1]) & (image[:, :, 2] == color[2])


def mask_to_image(mask):
    mask = np.tile(np.asarray(mask, dtype=np.int)[:, :, np.newaxis], reps=(1, 1, 3))
    image = np.ones_like(mask) * mask * 255
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)


def process_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3)))
    return mask


def detect_green_areas(image):
    image = cv2.resize(image, (960, 720))
    image = reduce_color_depth(image, 2)

    green_mask = extract_mask(image, (63, 191, 63))
    green_mask = mask_to_image(green_mask)
    green_mask = process_mask(green_mask)
    return green_mask


def discard_inner_ellipsoids(green_areas, center, lines):
    labeled_image, num_labels = label(green_areas > 127)
    # remove bull's
    bulls_labels = get_bulls_labels(center, green_areas, labeled_image)
    for bl in bulls_labels:
        green_areas[labeled_image == bl] = 0

    # turn lines clockwise a bit
    lines = np.copy(lines)

    # lines = np.asarray([[0.6, 3*np.pi/8]])
    lines = lines[:1]

    lines = rotate_lines(lines, np.pi/8, center)

    lines = convert_lines_to_absolute(green_areas, lines)
    # remove triple ring
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))

        cv2.line(green_areas, (x1, y1), (x2, y2), 255, 2)
    cv2.circle(green_areas, tuple((center * green_areas.shape[:2][::-1]).astype(np.int)), 5, 255, 5)
    cv2.imshow('title', green_areas)
    cv2.waitKey(0)


def rotate_lines(lines, angle, center):
    distances_before = calculate_point_line_distances(lines, center)
    lines[:, 1] += angle
    distances_after = calculate_point_line_distances(lines, center)
    # FIX: Either add OR subtract missing distance!
    lines[:, 0] += distances_after - distances_before
    return lines


def calculate_point_line_distances(lines, point):
    rhos, thetas = lines[:, 0], lines[:, 1]
    x = rhos * np.cos(thetas)
    y = rhos * np.sin(thetas)

    b = np.stack([-y, x], axis=-1)
    c = np.tile(point[np.newaxis, :], reps=(lines.shape[0], 1)) - np.stack([x, y], axis=-1)

    row_wise_scalar_product = np.sum(b * c, axis=-1)
    norms = np.linalg.norm(b, axis=-1) * np.linalg.norm(c, axis=-1)
    alphas = np.arccos(row_wise_scalar_product / norms)

    distances = np.sin(alphas) * np.linalg.norm(c, axis=-1)
    return distances


def get_bulls_labels(center, green_areas, labeled_image):
    DIST = 0.05
    h, w = green_areas.shape
    cx, cy = center
    bulls_labels = []
    for i in range(int((cx - DIST) * w), int((cx + DIST) * w)):
        for j in range(int(cy * h - DIST * w), int(cy * h + DIST * w)):
            if np.linalg.norm([i - cx * w, j - cy * h]) < DIST * w and labeled_image[i, j] > 0:
                bulls_labels.append(labeled_image[j, i])
    return bulls_labels


def f(image):
    lines = detect_lines(image)
    center = calculate_center(lines)

    green_areas = detect_green_areas(image)
    green_areas = discard_inner_ellipsoids(green_areas, center, lines)
    # corners = calculate_corners(image) # not clear yet how to do this reliably
    # a, b = select_opposing_starting_points(corners, center)
    # c, d = detect_perpendicular_points(green_areas, corners)
    # return a, b, c, d


def detect_lines(image):
    image = cv2.resize(image, (960, 720))
    image = cv2.blur(image, (3, 3))
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=200)

    rhos = lines[:, 0, 0]
    thetas = lines[:, 0, 1]
    lines = convert_lines_to_relative(image, rhos, thetas)
    return lines


def convert_lines_to_relative(image, rhos, thetas):
    h, w = image.shape[:2]
    lines = _scale_lines(h, w, rhos, thetas)
    return lines


def convert_lines_to_absolute(image, lines):
    rhos = lines[:, 0]
    thetas = lines[:, 1]

    h, w = image.shape[:2]
    h = 1 / h
    w = 1 / w

    lines = _scale_lines(h, w, rhos, thetas)
    return lines


def _scale_lines(h, w, rhos, thetas):
    x = rhos * np.cos(thetas)
    y = rhos * np.sin(thetas)
    t = (x * y * (w * w - h * h)) / (y * y * h * h + x * x * w * w)
    F1 = x / w + t * y / w
    F2 = y / h - t * x / h
    rhos = np.linalg.norm([F1, F2], axis=0)
    thetas = np.arctan2(F2, F1)
    lines = np.stack([rhos, thetas], axis=-1)
    return lines


def calculate_intersections(lines):
    intersections = []
    for line1, line2 in combinations(lines, 2):
        rho1, theta1 = line1
        rho2, theta2 = line2
        denominator = np.sin(theta1) * np.cos(theta2) - np.sin(theta2) * np.cos(theta1)
        if np.abs(denominator) < 0.001:
            # lines are (almost) parallel
            continue
        else:
            y = (rho1 * np.cos(theta2) - rho2 * np.cos(theta1)) / denominator
            x = (rho1 - y * np.sin(theta1)) / np.cos(theta1)
            # remove intersections outside of the image
            if 0 <= x < 1 and 0 <= y < 1:
                intersections.append((x, y))

    return intersections


def calculate_center(lines):
    pass
    # option 1:
    # center_estimate = calculate_nearest_point(lines)
    # lines_closest_to_estimate = select_closest_lines(lines, center_estimate, percentile)
    # return calculate_closest_point(lines_closest_to_estimate)

    # option 2:
    intersections = np.asarray(calculate_intersections(lines))
    clusters = DBSCAN(eps=0.01, min_samples=3).fit(intersections)
    occurrence_count = Counter(clusters.labels_)
    a, b = occurrence_count.most_common(2)[:][0]
    largest_cluster_index = a if a >= 0 else b
    return np.average(intersections[clusters.labels_ == largest_cluster_index], axis=0)


if __name__ == '__main__':
    image = cv2.imread('resources/board1.jpg', cv2.IMREAD_COLOR)

    f(image)

    # aligned = cv2.resize(image, (960, 720))
    # cv2.imshow('as', aligned)
    # cv2.waitKey(0)
