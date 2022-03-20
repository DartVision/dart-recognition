import cv2
import tensorflow as tf


class DartDetector(object):

    def __init__(self):
        self.detection_threshold = 0.5
        self.macro_detector = None
        self.micro_detector = None
        self.two_image_result_fusion = None
        self.coordinate_transform = None
        self._im_size = 300

    def detect(self, image1, image2):
        """
        Detects up to 3 darts on the given image pair.
        :param image1:
        :param image2:
        :return:
        """
        image_dimension = (self._im_size, self._im_size)
        resized_image1 = cv2.resize(image1, image_dimension, interpolation='bilinear')
        resized_image2 = cv2.resize(image2, image_dimension, interpolation='bilinear')
        images = tf.convert_to_tensor([resized_image1, resized_image2])

        detections = self.macro_detector(images)
        detections1 = detections[0]
        detections2 = detections[1]

        sub_images1 = self._extract_sub_image(image1, detections1)
        sub_images2 = self._extract_sub_image(image2, detections2)

        n = len(sub_images1)
        sub_images = tf.stack([sub_images1, sub_images2])

        sub_image_detections = self.micro_detector(sub_images)
        sub_image_detections1 = sub_image_detections[0:n]
        sub_image_detections2 = sub_image_detections[n:]

        transformed_detections1 = self.coordinate_transform(detections1, sub_image_detections1, image1)
        transformed_detections2 = self.coordinate_transform(detections2, sub_image_detections2, image2)

        detections = self.two_image_result_fusion(transformed_detections1, transformed_detections2)

        return detections

    def _extract_sub_image(self, image, detections):
        """
        Extracts regions of interest centered at the detected position
        :param image:
        :param detections:
        :return:
        """
        sub_images = []
        h, w = image.shape
        for detection in detections:
            if detections[0] >= self.detection_threshold:
                x, y = detection[2, 3]
                abs_x, abs_y = int(x * w), int(y * h)
                abs_x = max(min(abs_x, w - self._im_size // 2), self._im_size // 2)
                abs_y = max(min(abs_y, h - self._im_size // 2), self._im_size // 2)
                sub_image = image[abs_x - self._im_size // 2:abs_x + self._im_size / 2,
                            abs_y - self._im_size // 2: abs_y + self._im_size // 2]
                sub_images.append(sub_image)

        return sub_images
