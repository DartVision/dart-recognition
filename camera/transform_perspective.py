import cv2
import numpy as np
import json





if __name__ == '__main__':
    sample = cv2.imread("../resources/pi0-distorted-sample.jpg")
    with open('../resources/pi0-distorted-sample.json') as json_file:
        sample_board = json.load(json_file)
        transformed = transform_image(sample, sample_board["points"]["20L"], sample_board["points"]["6T"],
                                      sample_board["points"]["3R"], sample_board["points"]["8T"])
        cv2.imshow("distorted-and-transformed.jpg", transformed)
        cv2.waitKey()

