import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

MAX_FEATURES = 500
MATCH_THRESHOLD = 0.15


def align_images(base_image, image_to_align):
    image1 = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, desc1 = orb.detectAndCompute(image1, None)
    keypoints2, desc2 = orb.detectAndCompute(image2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(desc1, desc2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not bad matches
    num_good_matches = int(len(matches) * MATCH_THRESHOLD)
    matches = matches[:num_good_matches]

    # Draw top matches
    matches_on_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
    cv2.imshow("matches", matches_on_image)
    cv2.waitKey(0)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    homography, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography
    height, width, channels = base_image.shape
    image2_aligned = cv2.warpPerspective(image_to_align, homography, (width, height))

    return image2_aligned, homography


if __name__ == '__main__':
    image = cv2.imread('resources/board1.jpg', cv2.IMREAD_COLOR)
    image2 = cv2.imread('resources/board2.jpg', cv2.IMREAD_COLOR)
    image = cv2.resize(image, (960, 720))
    image2 = cv2.resize(image2, (960, 720))

    aligned, homography = align_images(image, image2)

    #  image = cv2.blur(image, (3, 3))
    # image = reduce_color_depth(image, 2)
    aligned = cv2.resize(aligned, (960, 720))
    cv2.imshow('as', aligned)
    cv2.waitKey(0)
