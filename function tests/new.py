import os
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from time import gmtime, strftime
from pyimagesearch import imutils

# Output images directory.
outputDir = "results"

# Specify a number which correspond to a receipt in the directory structure.
# numbers ight not always correspond to the same receipt, e.g. if we add a
# new directory of images, it will change the point in which some files are
# traversed.
# Useful for testing, e.g. prog 1 runs the pipline on image 1.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Integer: Image numbers (0 ..)")
args = vars(ap.parse_args())

# Grab all receipt jpgs and return a list of their paths.
def getReceipts(rootDir):
    images = []
    for dirName, _, fileList in os.walk(rootDir):
        print("Found directory: %s" % dirName)
        for fileName in fileList:
            if fileName.endswith("jpg"):
                print("\t%s" % fileName)
                path = dirName + "/" + fileName
                images.append(path)
    return images

def readReceipt(number):
    # Read from specificed path.
    image = cv2.imread(receipts[number])
    # Resize as the images are hi-res.
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height = 500)
    # Return image.
    return image

def generateFileName(number):
    date = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    return number + " -- " + date + ".jpg"

receipts = getReceipts(".")
receipt = readReceipt(int(args["image"]))
original = receipt.copy()

def nothing(x):
    pass

#image window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('th1','image',0,255,nothing)
cv2.createTrackbar('th2','image',0,255,nothing)
cv2.createTrackbar('x','image',1,100,nothing)
cv2.createTrackbar('y','image',1,100,nothing)

while(1):
    # get current positions of four trackbars
    th1 = cv2.getTrackbarPos('th1','image')
    th2 = cv2.getTrackbarPos('th2','image')
    x = cv2.getTrackbarPos('x','image')
    y = cv2.getTrackbarPos('y','image')
    data = "low: " + str(th1) + " , high: " + str(th2)
    print(data)
    #apply canny
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(receipt, (9,9), 0)
    edges = cv2.Canny(receipt, th1, th2, apertureSize=3)

    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morphKernel)

    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, morphKernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,4))
    edges = cv2.erode(edges, kernel)

    #show the image
    cv2.imshow('image',edges)
    #press ESC to stop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()



# # convert the image to grayscale, blur it, and find edges
# # in the image
# gray = cv2.cvtColor(receipt, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(gray, 75, 200)
#
# ###################
#
# (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
#
# # loop over the contours
# for c in cnts:
# 	# approximate the contour
# 	peri = cv2.arcLength(c, True)
# 	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#
# 	# if our approximated contour has four points, then we
# 	# can assume that we have found our screen
# 	if len(approx) == 4:
# 		screenCnt = approx
# 		break
#
# # show the contour (outline) of the piece of paper
# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
#
# ####################
#
# edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
#
# cv2.imshow("Results", np.hstack([original,edged]))
# k = cv2.waitKey(0)
# # wait for ESC key to exit
# if k == 27:
#     cv2.destroyAllWindows()
# # wait for 's' key to save and exit
# elif k == ord('s'):
#     cv2.imwrite(os.path.join(outputDir, generateFileName(args["image"])), receipt)
#     cv2.destroyAllWindows()
