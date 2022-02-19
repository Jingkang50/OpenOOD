import cv2
import numpy as np


def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


# opening morphological process for localization
def morphological_process(x):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel = kernel.astype(np.uint8)
    binary_map = x.astype(np.uint8)
    opening = cv2.morphologyEx(binary_map[0], cv2.MORPH_OPEN, kernel)
    opening = opening.reshape(1, opening.shape[0], opening.shape[1])
    for index in range(1, binary_map.shape[0]):
        temp = cv2.morphologyEx(binary_map[index], cv2.MORPH_OPEN, kernel)
        temp = temp.reshape(1, temp.shape[0], temp.shape[1])
        opening = np.concatenate((opening, temp), axis=0)
    return opening


def max_regarding_to_abs(a, b):
    c = np.zeros(a.shape)
    for i in range(len(a)):
        for j in range(len(a[0])):
            if abs(a[i][j]) >= abs(b[i][j]):
                c[i][j] = a[i][j]
            else:
                c[i][j] = b[i][j]
    return c
