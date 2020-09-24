import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


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
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((10, 10)))
    return mask


def fit_ideal_board_to_detection(ideal_points, detected_points):
    a, b, phi = 0, 0, 0

    return a, b, phi


if __name__ == '__main__':
    image = cv2.imread('resources/board1.jpg', cv2.IMREAD_COLOR)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    # image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)

    image = cv2.resize(image, (960, 720))
    image = cv2.blur(image, (3, 3))
    # image = reduce_color_depth(image, 2)
    # cv2.imshow('as', image)
    # cv2.waitKey(0)

    # compute masks and calculate intersections
    # red_mask = extract_mask(image, (63, 63, 191))
    # green_mask = extract_mask(image, (63, 191, 63))
    # white_mask = extract_mask(image, (191, 191, 191))
    # black_mask = extract_mask(image, (63, 63, 63))
    # red_mask = mask_to_image(red_mask)
    # green_mask = mask_to_image(green_mask)
    # white_mask = mask_to_image(white_mask)
    # black_mask = mask_to_image(black_mask)
    # red_mask = process_mask(red_mask)
    # green_mask = process_mask(green_mask)
    # white_mask = process_mask(white_mask)
    # black_mask = process_mask(black_mask)
    # edges_gr = cv2.bitwise_and(red_mask, green_mask)
    # edges_bw = cv2.bitwise_and(white_mask, black_mask)
    # edges = cv2.bitwise_and(edges_bw, edges_gr)
    #
    # result = edges
    #
    # green_mask = cv2.resize(green_mask, (1920, 1080))
    # cv2.imshow('asdf', green_mask)
    # cv2.waitKey(0)

    # result = mask_to_image(green_mask)
    # pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    # norm = colors.Normalize(vmin=-1., vmax=1.)
    # norm.autoscale(pixel_colors)
    # pixel_colors = norm(pixel_colors).tolist()

    # hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # h, s, v = cv2.split(hsv_image)
    # fig = plt.figure()
    # axis = fig.add_subplot(1, 1, 1, projection="3d")

    # axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    # axis.set_xlabel("Hue")
    # axis.set_ylabel("Saturation")
    # axis.set_zlabel("Value")
    # plt.show()

    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    result = edges

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    image = image[:, :, [2, 1, 0]]
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(result, cmap='gray')
    plt.show()
