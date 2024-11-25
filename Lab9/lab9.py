# import opencv
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

# Load the input image
image = cv2.imread('ATU1.jpg')
# Grayscaling the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Deep copying the original image
imgHarris = copy.deepcopy(image)

# Harris corner detection
dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)

#Plotting detected Harris corners
threshold = 0.05; #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(123, 55, 100),-1)

plt.subplot(2, 1,1),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 1,2),plt.imshow((imgHarris), cmap = 'gray')
plt.title('Harris corner detection'), plt.xticks([]), plt.yticks([])



plt.show()

