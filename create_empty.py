import cv2
import numpy as np


def create_empty():
    sq = np.zeros((100, 100, 4), np.uint8)
    cv2.imwrite("images/square.png", sq)


if __name__ == '__main__':
    create_empty()
