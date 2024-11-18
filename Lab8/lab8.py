# import opencv
import cv2
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10, 7))

rows = 2
columns = 2

# Load the input image
image = cv2.imread('ATU.jpg')
# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 3x3 blur
blurredimage =  cv2.GaussianBlur(gray_image,(3, 3),0)
# 13x13 blur
moreblurredimage = cv2.GaussianBlur(gray_image,(13, 13),0)

fig.add_subplot(rows, columns, 1) 
  
# showing image 
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.title("Original") 

# Adds a subplot at the 2nd position 
fig.add_subplot(rows, columns, 2) 
  
# showing image 
plt.imshow(cv2.cvtColor(gray_image,cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.title("Grayscale")

fig.add_subplot(rows, columns, 3) 
  
# showing image 
plt.imshow(cv2.cvtColor(blurredimage,cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.title("3x3 Blur")

fig.add_subplot(rows, columns, 4) 
  
# showing image 
plt.imshow(cv2.cvtColor(moreblurredimage,cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.title("13x13 Blur")

plt.show()

