# Image to Cartoon Converter
# Remember to download PIP, OpenCV, matplotlib to run this script.

# Importing modules
import cv2
print(cv2.__version__)
import numpy as np
import matplotlib.pyplot as plt

# Loading and plotting the original image: read the image using the imread function and then convert it to RGB format with the help of the cvtColor function.
# Then plot the image using the imshow function.
img = cv2.imread("dogimg.JPG")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis("off")
plt.title("JoJo at Mt. Hood - Original Image")
plt.show()

# Converting image to grayscale format using the cvtColor function.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray,(25,25),0)
# gray = cv2.medianBlur(gray, 5)
plt.figure(figsize=(10,10))
plt.imshow(gray,cmap="gray")
plt.axis("off")
plt.title("JoJo at Mt. Hood - Grayscale Image")
plt.show()

# Getting an edged image of the grayscale image and then applying the convolutional network to the image.
# The same is done by using the adaptiveThreshold function and setting the required parameters to get the edged image.
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
plt.figure(figsize=(10,10))
plt.imshow(edges,cmap="gray")
plt.axis("off")
plt.title("JoJo at Mt. Hood - Edged Image")
plt.show()

# Cartoonization: applying the convolutional filter using the bilateralFilter function. Then use the bitwise operation and pass the original image and the edged image to cartoonize the images.
color = cv2.bilateralFilter(img, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)
plt.figure(figsize=(10,10))
plt.imshow(cartoon,cmap="gray")
plt.axis("off")
plt.title("JoJo at Mt. Hood - Cartoon Image")
plt.show()

# Pencil Sketch: 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (25, 25), 0)
cartoon = cv2.divide(gray, gray_blur, scale=250.0)
plt.figure(figsize=(10,10))
plt.imshow(cartoon,cmap="gray")
plt.axis("off")
plt.title("JoJo at Mt. Hood - Pencil Sketch")
plt.show()

#The final output of turning images into cartoons.
# cv2.imshow("Image", img)
# cv2.imshow("edges", edges)
# cv2.imshow("Cartoon", cartoon)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


