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
image = cv2.imread(args["image"])
# Resize as the images are hi-res.
ratio = image.shape[0] / 500.0
# Keep a copy of original.
orig = image.copy()
# Resize image.
image = imutils.resize(image, height = 500)

image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY )


ret,thresh1 = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(image,100,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(image,100,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(image,100,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(image,100,255,cv2.THRESH_TOZERO_INV)

thresh = ['image','thresh1','thresh2','thresh3','thresh4','thresh5']

for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(eval(thresh[i]),'gray')
    plt.title(thresh[i])

plt.show()
