import cv2
import numpy as np
import argparse
from pyimagesearch import imutils


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

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,gray = cv2.threshold(gray,127,255,0)
gray2 = gray.copy()

contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if 200<cv2.contourArea(cnt)<5000:
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.rectangle(gray2,(x,y),(x+w,y+h),0,-1)

cv2.imshow('IMG',gray2)
cv2.waitKey(0)
cv2.destroyAllWindows()
