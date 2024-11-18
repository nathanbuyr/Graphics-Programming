# import opencv
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the input image
image = cv2.imread('ATU.jpg')
# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 3x3 blur
blurredimage =  cv2.GaussianBlur(gray_image,(3, 3),0)
# 13x13 blur
moreblurredimage = cv2.GaussianBlur(gray_image,(13, 13),0)
# Sobel images
sobelHorizontal = cv2.Sobel(blurredimage,cv2.CV_64F,1,0,ksize=5) # x dir
sobelVertical = cv2.Sobel(blurredimage,cv2.CV_64F,0,1,ksize=5) # y dir
sobel_added = sobelVertical + sobelVertical

plt.subplot(3, 3,1),plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3,3),plt.imshow(blurredimage, cmap = 'gray')
plt.title('3x3 Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3,4),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3,5),plt.imshow(sobelVertical, cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3,6),plt.imshow(sobel_added, cmap = 'gray')
plt.title('13x13 Blur'), plt.xticks([]), plt.yticks([])


plt.show()


