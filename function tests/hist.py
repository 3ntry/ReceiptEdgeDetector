from pyimagesearch import imutils
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# Read in image.
image = cv2.imread(args["image"], 0)
# Resize as the images are hi-res.
ratio = image.shape[0] / 500.0
# Keep a copy of original.
orig = image.copy()
# Resize image.
image = imutils.resize(image, height = 500)

image = cv2.GaussianBlur(image, (3, 3), 0)

print(image[400])

plt.figure(1)
plt.subplot(511)
plt.plot(image[100])

std_one = "100: " + str(np.std(image[100]))

print(std_one)

plt.subplot(512)
plt.plot(image[200])

std_two = "200: " + str(np.std(image[200]))

print(std_two)

plt.subplot(513)
plt.plot(image[300])

std_three = "300: " + str(np.std(image[300]))

print(std_three)

plt.subplot(514)
plt.plot(image[400])

plt.subplot(515)
plt.hist(image[400])

std_four = "400: " + str(np.std(image[400]))

print(std_four)

print(str(np.percentile(image[400], 10)))
print(str(np.percentile(image[400], 20)))
print(str(np.percentile(image[400], 30)))
print(str(np.percentile(image[400], 40)))
print(str(np.percentile(image[400], 50)))
print(str(np.percentile(image[400], 60)))
print(str(np.percentile(image[400], 70)))
print(str(np.percentile(image[400], 80)))
print(str(np.percentile(image[400], 90)))


cv2.line(image, (1,100), (400,100), (0, 255, 0), 1)
cv2.line(image, (1,200), (400,200), (0, 255, 0), 1)
cv2.line(image, (1,300), (400,300), (0, 255, 0), 1)
cv2.line(image, (1,400), (400,400), (0, 255, 0), 1)

cv2.namedWindow('Image')
cv2.imshow('Image', image)

plt.show()

# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([image],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()
