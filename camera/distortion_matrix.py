import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calc_distortion_matrix(chessboard_picture_path, pattern_size=(7, 9), interactive=True):
    """
    This procedure produces a camera matrix and distortion coefficients for a specific camera.
    As input, we expect a picture shot on the camera showing a standard checkerboard with the according patternSize
    (see https://google.com/search?q=chessboard+camera+calibration).
    Make sure, the chessboard is completely flat and well visible.
    :param interactive: interactive calibration
    """
    image = cv2.imread(chessboard_picture_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret_val, corners = cv2.findChessboardCorners(image, pattern_size, None)
    if not ret_val:
        raise ValueError("Could not find chessboard corners!")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    objpoints.append(objp)
    imgpoints.append(corners)

    if interactive:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(image, pattern_size, corners2, ret_val)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.setWindowTitle('img', 'Erkannte Ecken')
        cv2.imshow('img', image)
        cv2.waitKey(0)

    ret_val, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[:2], None, None)
    if not ret_val:
        raise ValueError("Could not calibrate camera.")

    if interactive:
        undistorted_image = undistort(image, mtx, dist)
        cv2.setWindowTitle('img', 'Nach Kalbrierung')
        cv2.imshow('img', undistorted_image)
        cv2.waitKey(0)
        cv2.destroyWindow('img')

    return mtx, dist


def undistort(distorted_img, mtx, dist):
    """
    Undistort image using the camera matrix and distortion coefficients
    obtained by the function calc_distortion_matrix
    """
    h, w = distorted_img.shape[0:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(distorted_img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y + h, x:x + w]
    return undistorted_img


if __name__ == '__main__':
    mtx0, dist0 = calc_distortion_matrix("../resources/pi0-chessboard.jpg", (7, 9), True)
    mtx1, dist1 = calc_distortion_matrix("../resources/pi1-chessboard.jpg", (7, 9), True)

    distorted_img = cv2.imread("../resources/pi0-distorted-sample.jpg")
    cv2.imwrite('pi0-undistorted-sample.jpg', undistort(distorted_img, mtx0, dist0))
