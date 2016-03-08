from pyimagesearch import imutils
import numpy as np
import argparse
import cv2
import pprint

def auto_canny(image, sigma = 0.7):
	# Compute median of single channel pixel intensities.
	v = np.median(image)
	# Apply automatic canndy edge detection using th ecomputed median.
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	# For info purposes
	print lower
	print upper
	edged = cv2.Canny(image, lower, upper, apertureSize = 3)
	# Return result
	return edged

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

# converting to greyscale just doe snot work!!
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Not sure on the fixed parameter (9,9). Seems to work on all so far though.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
#dilated = cv2.dilate(image, kernel)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)



cv2.imshow("Dilated", opening)
cv2.waitKey(4000)
cv2.destroyAllWindows()


th3 = cv2.adaptiveThreshold(opening,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

cv2.imshow("Adaptive threshold", th3)
cv2.waitKey(4000)
cv2.destroyAllWindows()

# normal_canny = cv2.Canny(opening, 75, 150, 3)
# auto_canny = auto_canny(opening)
#
# print "STEP 1: Edge Detection"
# cv2.imshow("Image", np.hstack([normal_canny, auto_canny]))
# cv2.waitKey(4000)
# cv2.destroyAllWindows()

minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(th3,1,np.pi/180,50,10,10)
#lines = cv2.HoughLines(auto_canny,1,np.pi/180,10)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("Image", image)
cv2.waitKey(5000)
cv2.destroyAllWindows()


# (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
#
# print cnts
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
# print "STEP 2: Find contours of paper"
# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()
