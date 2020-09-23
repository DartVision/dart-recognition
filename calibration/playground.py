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
    mask = np.tile(np.asarray(mask, dtype=np.int)[:, :, np.newaxis], reps=(1,1,3))
    image = np.ones_like(mask) * mask * 255
    return image


if __name__ == '__main__':
    image = cv2.imread('resources/board1.jpg', cv2.IMREAD_COLOR)
    image = cv2.resize(image, (960, 720))
    #  image = cv2.blur(image, (3, 3))
    image = reduce_color_depth(image, 2)
    # cv2.imshow('as', image)
    # cv2.waitKey(0)

    red_mask = extract_mask(image, (63, 63, 191))
    green_mask = extract_mask(image, (63, 191, 63))
    result = mask_to_image(red_mask)
    result = mask_to_image(green_mask)
    # pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    # norm = colors.Normalize(vmin=-1., vmax=1.)
    # norm.autoscale(pixel_colors)
    # pixel_colors = norm(pixel_colors).tolist()

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
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

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # red-green segmentation
    red1 = (1, 190, 200)
    red2 = (18, 255, 255)
    green1 = (30, 100, 50)
    green2 = (75, 255, 255)

    image = image[:, :, [2, 1, 0]]
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(result)
    plt.show()
