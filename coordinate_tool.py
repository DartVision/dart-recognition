import cv2
from os import path
import json


class CoordinateTool(object):
    def __init__(self, image_paths, directory):
        if len(image_paths) == 0:
            raise Exception('image_paths empty!')
        self._image_paths = image_paths
        self._current_image_path = None
        self._coordinates = []
        self._directory = directory
        cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Images', 2800, 2100)
        cv2.setMouseCallback('Images', self._mouse_callback)

    def start(self):
        i = 0
        while True:
            self._current_image_path = self._image_paths[i]
            json_path = path.join(self._directory,
                                  '.'.join(path.basename(self._image_paths[i]).split('.')[:-1]) + '.json')
            self._coordinates = self._load_coordinates(json_path)
            self._draw_image()
            key_pressed = cv2.waitKey(0)
            self._save_coordinates(json_path, self._coordinates)
            if key_pressed == 32:  # space
                i = (i + 1) % len(self._image_paths)
            elif key_pressed == ord('q'):
                cv2.destroyAllWindows()
                return

    def _mouse_callback(self, event, x, y, flags, param):
        w, h, _ = self._image.shape
        if event == cv2.EVENT_LBUTTONDOWN:
            for coordinate in self._coordinates:
                if abs(x - coordinate.x * w) <= 4 and (abs(y - coordinate.y * h)) <= 4:
                    self._coordinates.remove(coordinate)
                    return
            self._coordinates.append(Coordinate(x / w, y / h))
        self._draw_image()

    def _save_coordinates(self, file_path, coordinates):
        coordinate_list = [[c.x, c.y] for c in coordinates]
        with open(file_path, 'w') as file:
            json.dump({'coordinates': coordinate_list}, file)

    def _load_coordinates(self, file_path):
        if path.isfile(file_path):
            with open(file_path, 'r') as file:
                return [Coordinate(*c) for c in json.load(file)['coordinates']]
        else:
            return []

    def _draw_image(self):
        image = cv2.imread(self._current_image_path, cv2.IMREAD_COLOR)
        self._image = cv2.resize(image, (1600, 1200))
        w, h, _ = self._image.shape
        for coordinate in self._coordinates:
            cv2.circle(self._image, (int(coordinate.x * w), int(coordinate.y * h)), 3, (0, 0, 255),
                       thickness=cv2.FILLED)
        cv2.imshow('Images', self._image)


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y


if __name__ == '__main__':
    files = ['resources/board1.jpg', 'resources/board2.jpg']
    CoordinateTool(files, 'resources').start()
