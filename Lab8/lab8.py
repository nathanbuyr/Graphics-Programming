# import opencv
import cv2
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10, 7))

rows = 2
columns = 1

# Load the input image
image = cv2.imread('ATU.jpg')
# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fig.add_subplot(rows, columns, 1) 
  
# showing image 
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.title("First") 

# Adds a subplot at the 2nd position 
fig.add_subplot(rows, columns, 2) 
  
# showing image 
plt.imshow(cv2.cvtColor(gray_image,cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.title("Second")

plt.show()

