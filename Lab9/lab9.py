# import opencv
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

# Load the input image
image = cv2.imread('ATU1.jpg')
robimage = cv2.imread('rob.png')
# Grayscaling the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rob_gray = cv2.cvtColor(robimage, cv2.COLOR_BGR2GRAY)

#Deep copying the original image
imgHarris = copy.deepcopy(image)
robHarris = copy.deepcopy(robimage)

# Harris corner detection
dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)
robdst = cv2.cornerHarris(rob_gray, 2, 3, 0.04)

# Shi Tomasi algorithim
corners = cv2.goodFeaturesToTrack(gray_image,100,0.01,10)
robGFTT = cv2.goodFeaturesToTrack(rob_gray,100,0.01,10)

#Deep copying for Shi Tomasi
imgShiTomasi = copy.deepcopy(image)
robShitTomasi = copy.deepcopy(robimage)

# GFTT corner loop
for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(x,y),3,(0, 255, 0),-1)

for i in corners:
    x,y = i.ravel()
    cv2.circle(robShitTomasi,(x,y),3,(0, 255, 0),-1)

#Plotting detected Harris corners
threshold = 0.05; #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(123, 55, 100),-1)

for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(robHarris,(j,i),3,(123, 55, 100),-1)


# Initiate ORB detector
orb = cv2.ORB_create()
 
# find the keypoints with ORB
kp = orb.detect(gray_image,None)
robkp = orb.detect(rob_gray,None)
 
# compute the descriptors with ORB
kp, des = orb.compute(gray_image, kp)
robkp, robdes = orb.compute(rob_gray, robkp)
 
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(gray_image, kp, None, color=(0,255,0), flags=0)
robOrb = cv2.drawKeypoints(rob_gray, robkp, None, color=(0,255,0), flags=0)

plt.subplot(2, 2,1),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2,2),plt.imshow((imgHarris), cmap = 'gray')
plt.title('Harris corner detection'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2,3),plt.imshow((imgShiTomasi), cmap = 'gray')
plt.title('GFTT / Shi Tomasi corner detection'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2,4),plt.imshow((img2), cmap = 'gray')
plt.title('ORB detection'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)

plt.subplot(2, 2,1),plt.imshow(rob_gray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2,2),plt.imshow((robHarris), cmap = 'gray')
plt.title('Harris corner detection'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2,3),plt.imshow((robShitTomasi), cmap = 'gray')
plt.title('GFTT / Shi Tomasi corner detection'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2,4),plt.imshow((robOrb), cmap = 'gray')
plt.title('ORB detection'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)  

# Window shown waits for any key pressing event
cv2.destroyAllWindows()
