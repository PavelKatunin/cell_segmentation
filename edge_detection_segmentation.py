import cv2
from matplotlib import pyplot as plt
import math
import numpy as np


def edge_detection_segmentation(input_image):
    gray = cv2.cvtColor(input_image, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    result_image = np.zeros(shape=[image.shape[0], image.shape[1], 1], dtype=np.uint8)
    for row in range(sobel_x.shape[0]):
        for col in range(sobel_y.shape[1]):
            result_image[row, col] = math.sqrt(sobel_x[row, col, 0] ** 2 + sobel_y[row, col, 0] ** 2)
    thresh = cv2.adaptiveThreshold(result_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


image = cv2.imread('/Users/pavelkatunin/PycharmProjects/cell_segmentation/timelapse_1TO50.tif')
segmented = edge_detection_segmentation(image)

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(2, 2, 2)
plt.imshow(segmented, cmap='gray')
plt.title('Segmented')
plt.xticks([])
plt.yticks([])
plt.show()
