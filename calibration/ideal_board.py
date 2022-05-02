import numpy as np


def create_ideal_board():
    """
    Creates ideal board coordinates. Centered at [0, 0] with maximal radius of 1.
    Starts at line between 5 and 20
    :return:
    """
    # measurements taken from https://commons.wikimedia.org/wiki/File:Dartboard_Abmessungen.svg
    # all radiuses are measured to the inside edge of the wires
    eye_rad = 12.7 / 2
    bull_rad = 31.8 / 2
    triple_rad = 107
    triple_thickness = 8
    double_rad = 170
    double_thickness = 8
    num_segments = 20

    corners = []
    for i in range(num_segments):
        angle = get_angle(i, num_segments)
        rads = np.asarray(
            [eye_rad, bull_rad, triple_rad - triple_thickness, triple_rad, double_rad - double_thickness, double_rad])
        x = rads * np.cos(angle)
        y = rads * np.sin(angle)
        corners.extend(np.stack([x, y], axis=1))

    return np.asarray(corners) / double_rad


def get_angle(i, num_segments):
    # divide the circle into num_segments many segments of equal size and rotate by half a segment
    angle = 2 * np.pi * 1 / num_segments * (i - 0.5)
    return angle


if __name__ == '__main__':
    board = create_ideal_board()
    import matplotlib.pyplot as plt

    plt.plot(board[:, 0], board[:, 1], 'ro', markersize=2)
    for i, txt in enumerate(range(len(board))):
        plt.annotate(txt, board[i])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
