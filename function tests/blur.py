import os
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from time import gmtime, strftime
from pyimagesearch import imutils

# Output images directory. Needs to exist otherwise Python will complain.
outputDir = "../results"

# Arguments as a list of ints corresponding to receipt images.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True, nargs='+', type=int)
ap.add_argument('-r', "--resize", action='store_true', default=False)
args = vars(ap.parse_args())

# Finds all images, assumes all images are images of recipts.
def getReceipts(rootDir):
    for dirName, _, fileList in os.walk(rootDir):
        for fileName in fileList:
            yield dirName + "/" + fileName

# Given a path to an image. Read and resize it.
def readReceipt(path):
    image = cv2.imread(path)
    if (ap.parse_args().resize):
        image = imutils.resize(image, height = 500)
    return image

def nothing(x):
    pass

def generateFileName():
    numbers = " ".join(str(x) for x in args["images"])
    date = strftime("[ %Y-%m-%d (%H.%M.%S) ]", gmtime())
    return numbers + " " + date + ".jpg"

# Get list of receipts. String array of paths. Search from current directory.
receipts = [x for x in getReceipts("../") if x.endswith("jpg")]

# Filter by indicies which we want.
requestedReceipts = [receipts[i] for i in args["images"]]

# We can join images together for observing strategies on different recipts.
if len(requestedReceipts) == 1:
    image = readReceipt(requestedReceipts[0])
else:
    image = np.hstack([ readReceipt(x) for x in requestedReceipts ])

# Create image display window.
cv2.namedWindow('Image')

# create trackbars for color change
cv2.createTrackbar('kernel', 'Image', 1, 50, nothing)
#cv2.createTrackbar('iterations', 'Image', 1, 100, nothing)

while(1):
    # get current positions of four trackbars
    k = cv2.getTrackbarPos('kernel','Image')
    #i = cv2.getTrackbarPos('iterations','Image')

    if (k % 2 == 0): k += 1

    # Output current parameters in console.
    data = "k size: " + str(k)
    print(data)

    # Apply adaptive thresholding.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = cv2.GaussianBlur(gray, (k, k), 0)

    #show the image
    cv2.imshow('Image', result)
    #press ESC to stop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite(os.path.join(outputDir, generateFileName()), result)
        break

cv2.destroyAllWindows()
