import cv2
import matplotlib.pyplot as plt
import json
import numpy as np

from calibration.ideal_board import create_ideal_board

if __name__ == '__main__':
    with open('resources/board1.json') as file:
        coordinates = np.asarray(json.load(file)['coordinates'])

    x = coordinates[:, 0]
    y = coordinates[:, 1] * -1 + 1

    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.plot(x, y, 'bo', markersize=2)
    for i, txt in enumerate(range(len(coordinates))):
        ax.annotate(txt, (x[i], y[i]))

    ideal_board = create_ideal_board()
    src_points = coordinates[[0, 49, 74, 55]]
    dest_points = ideal_board[[101, 41, 71, 11]]
    # ax = fig.add_subplot(212)
    # ax.plot(x, y_new, 'bo')
    M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dest_points.astype(np.float32))

    transformed_coords = np.array([coordinates])
    transformed_coords = cv2.transform(transformed_coords, M)

    ax = fig.add_subplot(212)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    z = transformed_coords[0, :, 2]
    ax.plot(transformed_coords[0, :, 0]/z, transformed_coords[0, :, 1]/z, 'bo', markersize=2)
    plt.show()

    with open('resources/board1.json') as file:
        coordinates = np.asarray(json.load(file)['coordinates'])

    src_points = coordinates[[0, 49, 74, 55]]
    dest_points = (ideal_board[[101, 41, 71, 11]] + np.ones((4, 2))) / 2
    dest_points[:, 1] = dest_points[:, 1] * -1 + 1
    src_points = src_points.astype(np.float32)
    dest_points = dest_points.astype(np.float32)
    src_points = (src_points * [640, 480]).astype(np.int).astype(np.float32)
    dest_points = (dest_points * [400, 400]).astype(np.int).astype(np.float32)
    dest_points[:, 0] += 100
    dest_points[:, 1] += 20
    M = cv2.getPerspectiveTransform(src_points, dest_points)

    orig = cv2.imread('resources/board1.jpg', cv2.IMREAD_COLOR)
    orig = cv2.resize(orig, (640, 480))
    warped = cv2.warpPerspective(orig, M, (600, 600))
    cv2.imshow('Im', orig)
    cv2.imshow('Warped', warped)
    cv2.waitKey()
