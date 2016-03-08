from pyimagesearch import imutils
import numpy as np
import argparse
import cv2
import pprint

def auto_canny(image, sigma = 0.33):
	# Compute median of single channel pixel intensities.
	v = np.median(image)
	# Apply automatic canndy edge detection using th ecomputed median.
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	print lower
	print upper
	edged = cv2.Canny(image, lower, upper)
	# Return result
	return edged

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Image", gray)
# cv2.waitKey(3000)
# cv2.destroyAllWindows()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
dilated = cv2.dilate(gray, kernel)

cv2.imshow("Image", dilated)
cv2.waitKey(3000)
cv2.destroyAllWindows()

th3 = cv2.adaptiveThreshold(dilated,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,0)

cv2.imshow("Image", th3)
cv2.waitKey(3000)
cv2.destroyAllWindows()

# normal_canny = cv2.Canny(image, 10, 200, 2)
# auto_canny = auto_canny(image)
#
# cv2.imshow("Image", np.hstack([normal_canny, auto_canny]))
# cv2.waitKey(5000)
# cv2.destroyAllWindows()
#
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(auto_canny,1,np.pi/180,50,10,10)
# #lines = cv2.HoughLines(auto_canny,1,np.pi/180,10)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imshow("Image", image)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()



#
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12,12))
# dilated = cv2.dilate(gray, kernel)
#
# th3 = cv2.adaptiveThreshold(dilated,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,2)
# #th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
#
# #blur = cv2.GaussianBlur(image,(5,5),0)
# #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# cv2.imshow("Dilated", th3)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()
#
# normal_canny = cv2.Canny(th3, 75, 150, 3)
# auto_canny = auto_canny(th3)
#
# print "STEP 1: Edge Detection"
# cv2.imshow("Image", np.hstack([normal_canny, auto_canny]))
# cv2.waitKey(5000)
# cv2.destroyAllWindows()
#
# minLineLength = 100
# maxLineGap = 20
# lines = cv2.HoughLinesP(auto_canny,1,np.pi/180,100,minLineLength,maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imshow("Image", image)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()
