# import opencv
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the input image
image = cv2.imread('ATU1.jpg')
# Grayscaling the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(2, 1,1),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.show()

