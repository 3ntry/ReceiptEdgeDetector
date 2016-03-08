import sys, cv2
from pyimagesearch import imutils

# The program accepts one command line parameter, specifying the file to read.
img = cv2.imread(sys.argv[1])

ratio = img.shape[0] / 500.0
orig = img.copy()
img = imutils.resize(img, height = 500)

## CROPPING AND DESKEWING
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#blurred = cv2.GaussianBlur(gray, (9,9), 0)

edges = cv2.Canny(img, 0, 150, apertureSize=3)

cv2.imshow("Image", edges)
cv2.waitKey(3000)
cv2.destroyAllWindows()

morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morphKernel)

cv2.imshow("Image", edges)
cv2.waitKey(3000)
cv2.destroyAllWindows()

morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, morphKernel)

cv2.imshow("Image", edges)
cv2.waitKey(3000)
cv2.destroyAllWindows()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,4))
edges = cv2.erode(edges, kernel)

cv2.imshow("Image", edges)
cv2.waitKey(3000)
cv2.destroyAllWindows()
