import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from matplotlib import colors
from sklearn.cluster import DBSCAN
from collections import Counter


def f(image):
    pass
    lines = detect_lines(image)
    center = calculate_center(lines)
    # for inters in center:
    #     cv2.circle(image, tuple((np.asarray(inters) * (image.shape[:2][::-1])).astype(np.int)), 10, (0, 255, 0), thickness=15)
    cv2.circle(image, tuple((np.asarray(center) * (image.shape[:2][::-1])).astype(np.int)), 10, (0, 255, 0), thickness=10)
    image = cv2.resize(image, (960, 720))
    cv2.imshow('as', image)
    cv2.waitKey(0)

    # green_areas = detect_green_areas(image)
    # green_areas = discard_inner_ellipsoids(green_areas, center)
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
    x = rhos * np.cos(thetas)
    y = rhos * np.sin(thetas)
    h, w = image.shape[:2]

    t = (x * y * (w*w - h*h)) / (y*y*h*h + x*x*w*w)
    F1 = x / w + t * y / w
    F2 = y / h - t * x / h

    rhos = np.linalg.norm([F1, F2], axis=0)
    thetas = np.arctan2(F2, F1)

    return np.stack([rhos, thetas], axis=-1)


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
            if 0 <=x < 1 and 0 <=y < 1:
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
    # return np.average(filtered_intersections, axis=0)


if __name__ == '__main__':
    image = cv2.imread('resources/board1.jpg', cv2.IMREAD_COLOR)

    f(image)

    # aligned = cv2.resize(image, (960, 720))
    # cv2.imshow('as', aligned)
    # cv2.waitKey(0)
