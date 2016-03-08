import sys, cv2, numpy as np, time
from matplotlib import cm, pyplot as plt

# CONSTANTS
# The interval between each cross-section used to calculate the
# boundaries.
INTERVAL = 10
# The width and height of the resized image.
WIDTH = 300
HEIGHT = 400

def pipeline(image):
	# We start timing the pipeline's execution time.
	start_time = time.time()
	# The program accepts one command line parameter, specifying the file
	# to read. We read the image in grayscale, and shrink it to reduce the
	# cost of the edge-detection algorithm
	img = cv2.imread(image, 0)
	img = cv2.resize(img, (WIDTH, HEIGHT))

	# We only examine every tenth cross-section, in order to increase the
	# algorithm's speed.
	crossSectionHeights = xrange(0, HEIGHT, INTERVAL)

	# We find the best boundary on the left-hand side of the image.
	boundariesLeft = findBoundaryPoints(img, crossSectionHeights, True)
	connectingLinesLeft = findLineEquations(boundariesLeft)
	bestBoundaryLeft = maxScoreLine(connectingLinesLeft, boundariesLeft)
	# We then do the same for the right-hand side of the image.
	boundariesRight = findBoundaryPoints(img, crossSectionHeights, False)
	connectingLinesRight = findLineEquations(boundariesRight)
	bestBoundaryRight = maxScoreLine(connectingLinesRight, boundariesRight)

	# We find the slope of the top and bottom borders.
	shortEdgeSlope = findShortEdgeSlope(bestBoundaryLeft[0], bestBoundaryRight[0])
	# We find the highest and lowest edge heights for both sides of the
	# receipt...
	leftShortEdges = findShortEdgeHeights(boundariesLeft, bestBoundaryLeft)
	rightShortEdges = findShortEdgeHeights(boundariesRight, bestBoundaryRight)
	# ...and then take the highest and lowest overall...
	topEdgePoint = min(leftShortEdges[0], rightShortEdges[0], key = lambda tup: tup[1])
	bottomEdgePoint = max(leftShortEdges[1], rightShortEdges[1], key = lambda tup: tup[1])
	# ..and draw a line through those points with the slope calculated above.
	topEdgeCoords = findShortEdge(topEdgePoint, shortEdgeSlope)
	bottomEdgeCoords = findShortEdge(bottomEdgePoint, shortEdgeSlope)

	# We print the execution time of the pipeline (excluding the
	# drawing time).
	print("--- %2s seconds ---" % (round(time.time() - start_time, 2)))

	# We draw the grayscale image.
	plt.imshow(img, cmap = cm.Greys_r)
	# We draw the boundary points onto the image, for evaluation purposes.
	scatterXsLeft = [boundary for (boundary, height) in boundariesLeft]
	scatterYsLeft = [height for (boundary, height) in boundariesLeft]
	scatterXsRight = [boundary for (boundary, height) in boundariesRight]
	scatterYsRight = [height for (boundary, height) in boundariesRight]
	plt.scatter(scatterXsLeft, scatterYsLeft)
	plt.scatter(scatterXsRight, scatterYsRight)
	# We also draw on the detected receipt boundaries.
	drawBoundary(img, bestBoundaryLeft)
	drawBoundary(img, bestBoundaryRight)
	cv2.line(img, topEdgeCoords[0], topEdgeCoords[1], 1, 2)
	cv2.line(img, bottomEdgeCoords[0], bottomEdgeCoords[1], 1, 2)
	# We display the final results.
	plt.show()

# This function moves leftwards/rightwards along a cross-section of
# the receipt, and returns the first pixel that is 1/2 darker than
# the current pixel (unless that pixel is too close to the center, in
# which case it is considered noise and None is returned).
def findCutoff(startX, height, image, left):
	# We extract the cross-section we will be working with from the image,
	# and intialize the starting x position.
	line, currentX = image[height], startX

	# The function can traverse left or right.
	if left == True:
		# While there are still pixels to traverse...
		while currentX > 1:
			# The int() cast is neccessary because the pixels values are
			# ints (i.e. max 255).
			currentDarkness = int(line[currentX])
			for skip in [1, 2]:
				boundaryX = currentX - skip
				# If the pixel darkness suddenly jumps by over 15...
				if abs(currentDarkness - line[boundaryX]) > 15:
					# Any pixels too close to the center of the image are
					# considered noise, and excluded. They usually
					# correspond to text lines of the receipt, and thus
					# won't usually help in picking up receipt boundaries.
					if boundaryX > (WIDTH * 3/8):
						return None
					else:
						return (int(boundaryX), height)
			else:
				currentX -= 1

	# This branch follows an almost identical logic to 'left == True'.
	elif left == False:
		while currentX < (WIDTH - 10):
			currentDarkness = int(line[currentX])
			for skip in [1, 2]:
				boundaryX = currentX + skip
				if abs(currentDarkness - line[boundaryX]) > 15:
					if boundaryX < (WIDTH * 5/8):
						return None
					else:
						return (int(boundaryX), height)
			else:
				currentX += 1

# Given an array of y-axes and a direction, this function returns all
# the corresponding boundary points.
def findBoundaryPoints(img, heights, left):
	boundaries = []
	for height in heights:
		boundary = findCutoff(WIDTH * 0.5, height, img, left)
		if boundary != None:
			boundaries.append(boundary)

	return boundaries

# This function calculates the equation of the lines connecting each
# pair of boundary points.
def findLineEquations(points):
	lineEquations = []
	for i, p1 in enumerate(points):
		for p2 in points[i+1:]:
			# If the line has a vertical slope, we cannot calculate
			# its equation. So we store it differently, with the
			# x-value as the intercept.
			if p2[0] == p1[0]:
				slope = float('inf')
				intercept = p1[0]
			# Otherwise, we store the line's equation, where the
			# intercept is y's value when x equals 0.
			else:
				slope = float(p1[1] - p2[1]) / (p1[0] - p2[0])
				intercept = p1[1] - (p1[0] * slope)
			lineEquations.append((slope, intercept))

	return lineEquations

# This function returns the highest-scoring line, which (hopefully)
# corresponds to the receipt boundary. The lines are scored based on
# the number of points less than 10/5 x-pixels away.
def maxScoreLine(lines, points):
	# We store the highest scoring line (i.e. the line of best fit).
	bestLine, hiScore = None, 0

	for p1 in lines:
		score = 0
		for p2 in points:
			if p1[0] == float('inf'):
				dist = abs(p1[1] - p2[0])
			else:
				xVal = (p2[1] - p1[1]) / p1[0]
				dist = abs(p2[0] - xVal)
			if dist < 10:
				score += 1
				if dist < 5:
					score += 1
		if score > hiScore:
			bestLine, hiScore = p1, score

	return bestLine

# This function draws the boundary on the image based on its equation.
# It handles the special case of the line having no slope.
def drawBoundary(img, boundary):
	if boundary[0] == float('inf'):
		boundaryP1 = (boundary[1], 0)
		boundaryP2 = (boundary[1], HEIGHT)
	else:
		# Previously, we used the endpoints were x = 0 and x = 300.
		# However, cv2.line() could not handle this approach for
		# almost vertical lines (ones which led to very high y-values).
		# Now, we use the endpoints where y = 0 and y = 400.
		startX = (0 - boundary[1]) / boundary[0]
		endX = (HEIGHT - boundary[1]) / boundary[0]
		boundaryP1 = (int(startX), 0)
		boundaryP2 = (int(endX), HEIGHT)
	cv2.line(img, boundaryP1, boundaryP2, 1, 2)

# This function returns the predicted slope of the top and bottom
# edges.
def findShortEdgeSlope(leftBoundarySlope, rightBoundarySlope):
	# If the two edges point in opposite directions (one left to
	# right, and the other right to left), then the receipt is
	# likely foreshortened rather than skewed, and a horizontal top
	# is the best approach.
	if leftBoundarySlope * rightBoundarySlope < 0:
		return 0
	# If one of the long edges is vertical, the slope of the top edge
	# will be equal to zero.
	elif leftBoundarySlope == float('inf') or rightBoundarySlope == float('inf'):
		return 0
	else:
		# It is my understanding that if the two long edges are not
		# parallel, the error is probably with the more skewed edge).
		if abs(leftBoundarySlope) < abs(rightBoundarySlope):
			return -1 / leftBoundarySlope
		else:
			return -1 / rightBoundarySlope

# This function finds the highest and lowest boundary points that are
# on the best-fit boundary lines. The top and bottom edges of the
# receipt will pass through these.
def findShortEdgeHeights(boundaries, bestBoundary):
	top = None

	if bestBoundary[0] == float('inf'):
		for (x, y) in boundaries:
			# The input array is ordered high-to-low, so for the top edge
			# we want the first boundary point to fulfill the condition.
			if top == None and abs(x - bestBoundary[1]) < 10:
				top = (x, y)
			# Whereas for the bottom edge, we want the lowest point
			# possible.
			if abs(x - bestBoundary[1]) < 5:
				bottom = (x, y)

	else:
		for (x, y) in boundaries:
			boundaryLineX = (y - bestBoundary[1]) / bestBoundary[0]
			if top == None and abs(x - boundaryLineX) < 10:
				top = (x, y)
			if abs(x - boundaryLineX) < 5:
				bottom = (x, y)

	return (top, bottom)

# This function finds the coordinates of the short edges based on a
# point and a slope.
def findShortEdge(point, slope):
	intercept = point[1] - (point[0] * slope)
	edgeStart = (0, int(intercept))
	edgeEnd = (WIDTH, int(intercept + slope * WIDTH))
	return (edgeStart, edgeEnd)

pipeline(sys.argv[1])
