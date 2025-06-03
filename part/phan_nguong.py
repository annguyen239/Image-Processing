import os
import cv2 as cv
import numpy as np

def phan_nguong(img):
    # If image has 3 dimensions (color), convert it to grayscale
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    M, N = img.shape
    imgout = np.zeros((M, N), np.uint8)
    for i in range(M):
        for j in range(N):
            r = img[i, j]
            if r == 63:
                s = 255
            else:
                s = 0
            imgout[i, j] = np.uint8(s)
    imgout = cv.medianBlur(imgout, 7)
    return imgout

