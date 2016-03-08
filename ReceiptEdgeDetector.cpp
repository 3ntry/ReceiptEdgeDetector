/*

Methods for generating points:

- Totally random.
- Generate a bunch of pre-selected points, jumble them up them up and then
  pop off a list for each new line. They are generated to reduce overlap.
	Randomness helps to ensure that it wont fail on the same image every time.

Heuristics for keeping / throwing away edges:

- First edge detected is probably the actual one BUT make it harder to detect
  edges for the first few patches.

Other to do:

- Terminate edge finding after first edge found.
- Deal with infinite gradients.
- Determine all line gradients and cluster based on gradient + proximity on
  x or y axis. This should put lines for a particular edge of the receipt in
	one cluster. Random edges found will be ditched.
- Then, cluster perpendicular sets of clustered lines. then we'll have all lines
  around the receipt. Ditch all others.
- As a result of the above. Build contour around receipt.

*/

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <math.h>
#include <random>
#include <algorithm>
#include <string>

using namespace cv;
using namespace std;

const unsigned int WIDTH = 450;
const unsigned int HEIGHT = 600;

const Point CENTRE = Point(WIDTH / 2, HEIGHT / 2);

const unsigned int DILATION_KERNEL_SIZE = 3;
const unsigned int DILATION_ITERATIONS = 3;

const unsigned int PATCH_WIDTH = 30;
const unsigned int PATCH_HEIGHT = 30;

// Probably using too many points.
const Point pts[] = {
	// Left-hand side.
	Point(0,0),Point(0,20),Point(0,40),Point(0,60),
	Point(0,80),Point(0,100),Point(0,120),Point(0,140),
	Point(0,160),Point(0,180),Point(0,200),Point(0,220),
	Point(0,240),Point(0,260),Point(0,280),Point(0,300),
	Point(0,320),Point(0,340),Point(0,360),Point(0,380),
	Point(0,400),Point(0,420),Point(0,440),Point(0,460),
	Point(0,480),Point(0,500),Point(0,520),Point(0,540),
	Point(0,560),Point(0,580),Point(0,600),
	// Right-hand side.
	Point(450,0),Point(450,20),Point(450,40),Point(450,60),
	Point(450,80),Point(450,100),Point(450,120),Point(450,140),
	Point(450,160),Point(450,180),Point(450,200),Point(450,220),
	Point(450,240),Point(450,260),Point(450,280),Point(450,300),
	Point(450,320),Point(450,340),Point(450,360),Point(450,380),
	Point(450,400),Point(450,420),Point(450,440),Point(450,460),
	Point(450,480),Point(450,500),Point(450,520),Point(450,540),
	Point(450,560),Point(450,580),Point(450,600),
	// Top.
	Point(20,0),Point(20,0),Point(40,0),Point(60,0),
	Point(80,0),Point(100,0),Point(120,0),Point(140,0),
	Point(160,0),Point(180,0),Point(200,0),Point(220,0),
	Point(240,0),Point(260,0),Point(280,0),Point(300,0),
	Point(320,0),Point(340,0),Point(360,0),Point(380,0),
	Point(400,0),Point(420,0),Point(440,0),
	// Bottom.
	Point(20,600),Point(20,600),Point(40,600),Point(60,600),
	Point(80,600),Point(100,600),Point(120,600),Point(140,600),
	Point(160,600),Point(180,600),Point(200,600),Point(220,600),
	Point(240,600),Point(260,600),Point(280,600),Point(300,600),
	Point(320,600),Point(340,600),Point(360,600),Point(380,600),
	Point(400,600),Point(420,600),Point(440,600)
};

// Line points and gradient.
struct Line {

	// Public struct members.
	Point pt1;
	Point pt2;
	double gradient;

	// Constructor.
	Line(const Point& a, const Point& b, double g) :
		pt1(a), pt2(b), gradient(g) {
	}

	// For debugging.
	string toString() {
		return "Line [(" + to_string(pt1.x) + "," + to_string(pt1.y) + "),(" +
			to_string(pt2.x) + "," + to_string(pt2.y) + ")]\t" + to_string(gradient);
	}

	// For sorting, prior to clustering.
	bool operator<(const Line& l) const {
			return gradient < l.gradient;
	}
};





// Generate a random number [low high]
int rand(unsigned int low, unsigned int high) {
	// From stackoverflow.
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> dis(low, nextafter(high + 1, DBL_MAX));
	return (int) dis(gen);
}

// Coin flipper function.
bool coinFlip() {
	return rand(1, 2) % 2 == 0;
}

// Generate a random point on the edge of the image.
// Depreciated but useful for testing.
Point generateRandomPoint() {
	if (coinFlip()) {
		unsigned int h = rand(1, HEIGHT);
		if (coinFlip()) {
			return Point(0, h);
		}
		return Point(WIDTH, h);
	}
	unsigned int w = rand(1, WIDTH);
	if (coinFlip()) {
		return Point(w, 0);
	}
	return Point(w, HEIGHT);
}

// We want to traverse lines from centre to edge.
bool getDirection(const Point& p) {
	if (p.x < CENTRE.x) {
		return false;
	}
	return true;
}

// Grab the patch from the source image.
void getMatrixSubset(const Mat& source, Mat& dest, const Point& p, const Size& s) {
	source(Rect(p.x, p.y, s.width, s.height)).copyTo(dest);
}

// Given two points of a line, calculate the gradient.
double calculateGradient(const Point& a, const Point& b) {
	return (((double) b.y - (double) a.y) / ((double) b.x - (double) a.x));
}

// Perform adaptive threshold and then probabilistic hough lines.
// Return the lines.
void processATHLPLines(const Mat& input, const Point& origin, vector<Line>& lines) {
	Mat thresholdedPatch;
	adaptiveThreshold(input, thresholdedPatch, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3, -1);

	// Hough lines.
	vector<Vec4i> houghLines;
	// The parameters here are quite sensitive.
	HoughLinesP(thresholdedPatch, houghLines, 1, CV_PI/180, 15, 20, 3);

	// Calculate gradient and global points for each line.
	for(size_t i = 0; i < houghLines.size(); i++) {

		// Calculate global points.
		Point a = Point(origin.x + houghLines[i][0], origin.y + houghLines[i][1]);
		Point b = Point(origin.x + houghLines[i][2], origin.y + houghLines[i][3]);

		// Calculate gradient.
		double g = calculateGradient(a, b);

		// Create new line.
		Line l(a, b, g);

		// Add to lines.
		lines.push_back(l);
	}
}

// Perform adaptive threshold and then probabilistic hough lines.
// Don't return lines, draw them instead.
// For testing.
void processATHLP(const Mat& input, Mat& original, const Point& origin) {
	Mat thresholdedPatch;
	adaptiveThreshold(input, thresholdedPatch, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3, -1);

	// Hough lines.
	vector<Vec4i> lines;
	// The parameters here are quite sensitive.
	HoughLinesP(thresholdedPatch, lines, 1, CV_PI/180, 15, 20, 3);

	//Create a copy to draw a coloured line on.
	Mat copy;
	cvtColor(thresholdedPatch, copy, CV_GRAY2RGB);

	// Draw lines on an image.
	for(size_t i = 0; i < lines.size(); i++) {
		Point p = Point(origin.x + lines[i][0], origin.y + lines[i][1]);
		Point q = Point(origin.x + lines[i][2], origin.y + lines[i][3]);

		// Draw the lines. Change the colour Green (100,255,0) / White.
		line(original, p, q, Scalar(100,255,0), 1, 8);
	}

	Mat resized;
	resize(copy, resized, Size(300, 300), 0, 0, INTER_AREA);
	imshow("Adaptive THreshold -> HoughLinesP", resized);
}

int main(int argc, char **argv) {
	// Read input image.
	Mat original = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	// Check if valid file.
	if(original.empty()) {
		cout <<  "Could not open or find the image." << endl;
		return -1;
	}

	// Don't need high-resolution photos for edge detection.
	Mat resized;
	resize(original, resized, Size(WIDTH, HEIGHT), 0, 0, INTER_AREA);

	// Colour copy for drawing coloured things.
	Mat copy;
	cvtColor(resized, copy, CV_GRAY2RGB);

	// A zereod matrix for just drawing features.
	Mat zeros(resized.size(), CV_8U);

	// Dilate.
	Mat dilated;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(resized, dilated, kernel, Point(-1, -1), 3);

	// Make points vector. Can't initialise vector with initialiser list.
	vector<Point> points(pts, &pts[sizeof(pts)/sizeof(pts[0])]);

	// Shuffle pre-defined points. Don't want deterministic behaviour.
	// NOTE. PROBABLY NEED TO SEED THIS.
	random_shuffle(points.begin(), points.end());

	// Vector to hold the lines which mark the edges we find.
	vector<Line> lines;

  // Grab each point.
  for (vector<Point>::iterator it = points.begin(); it != points.end(); ++it) {

		// Get a point.
		Point p = (Point) *it;

		// Draw line. For testing.
		//line(copy, CENTRE, p, Scalar(100,255,0), 1, 8);

		// Return an iterator to all pixels on the line, p.
		LineIterator pixels(dilated, CENTRE, p, 8, getDirection(p));

		// Iterate through line pixels.
		for(int i = 0; i < pixels.count; i++, pixels++) {

			// We only want every 20th pixel.
	    if (i % 20 != 0) continue;

			// Two points to define the patch rectangle.
			Point pt1 = pixels.pos();
			Point pt2(pt1.x + PATCH_WIDTH, pt1.y + PATCH_HEIGHT);

			// Reduce patch size if right hand edge side > WIDTH.
			if (pt1.x >= WIDTH - PATCH_WIDTH) {
				pt2.x -= PATCH_WIDTH - (WIDTH - pt1.x);
			}

			// As above but for HEIGHT.
			if (pt1.y >= HEIGHT - 20) {
				pt2.y -= PATCH_HEIGHT - (HEIGHT - pt1.y);
			}

			// Draw where we are currently looking. For testing.
			//rectangle(copy, pt1, pt2, Scalar(100, 255, 0), CV_FILLED, 8, 0);

			// Get the patch data.
			Mat subMatrix;
			getMatrixSubset(dilated, subMatrix, pt1, Size(abs(pt1.x - pt2.x), abs(pt1.y - pt2.y)));

			// Process.
			// Don't return lines - see them instead.
			//processATHLP(subMatrix, copy, pt1);
			// Populate lines vector with foudn lines..
			processATHLPLines(subMatrix, pt1, lines);

			// Pause and update image after each patch is processed.
			// imshow("Display window", copy);
			// int k = waitKey(0);
			// if(k == 27) {
			// 	exit(0);
			// }
		}
	}

	// sort(lines.begin(), lines.end());

	// Bucket lines.
	vector<Line> shallow;
	vector<Line> steep;
	for(Line l : lines) {
		if (abs(l.gradient) > 2) {
			steep.push_back(l);
		} else {
			shallow.push_back(l);
		}
	}

	for (auto l : steep) {
		Point p(0,0), q(copy.size().width,copy.size().height);
		p.y = -(l.pt1.x - p.x) * l.gradient + l.pt1.y;
		q.y = -(l.pt2.x - q.x) * l.gradient + l.pt2.y;
		line(copy, p, q, Scalar(100,255,0), 1, 8);
	}

	for (auto l : shallow) {
		Point p(0,0), q(copy.size().width,copy.size().height);
		p.y = -(l.pt1.x - p.x) * l.gradient + l.pt1.y;
		q.y = -(l.pt2.x - q.x) * l.gradient + l.pt2.y;
		line(copy, p, q, Scalar(0,100,255), 1, 8);
	}



	// Show results.
	// imshow("Display window", copy);
	// //imshow("Display window", drawing);
	// int k = waitKey(0);
	// if(k == 27) {
	// 	exit(0);
	// }

	// OR Write results to /results directory.
	string str = argv[1];
	string fileName = str.substr(7, str.length());
	imwrite("results/" + fileName, copy);

	return 0;
}



//
// #include <opencv2/opencv.hpp>
// #include <vector>
// #include <string>
// #include <math.h>
// #include <stack>
// #include <random>
//
// using namespace cv;
// using namespace std;
//
// // Size of image we wish to operate on. Ratio 0.75.
// const unsigned int WIDTH = 450;
// const unsigned int HEIGHT = 600;
//
// // Dilation parameters.
// const unsigned int DILATION_KERNEL_SIZE = 3;
// const unsigned int DILATION_ITERATIONS = 3;
//
// // Patch dimentions.
// const unsigned int PATCH_WIDTH = 30;
// const unsigned int PATCH_HEIGHT = 30;
//
// // Possible directions.
// enum Direction {
// 	UP = 0,
// 	RIGHT = 1,
// 	DOWN = 2,
// 	LEFT = 3
// };
//
// // Patch class.
// class Patch {
//
// private:
//
// 	const Mat& image;									// Const ptr to the original image.
// 	Point origin;											// Origin of current patch (top left).
// 	unsigned int width;								// Width of path.
// 	unsigned int height;							// Height of patch.
// 	Mat data;													// The patch data.
// 	stack<Point> moves;								// Stack for next possible moves.
// 	int directions[4] = {0, 0, 0 ,0}; // Record of directions travelled.
// 	int currentDirection;							// The current direction;
//
// 	// Pull the patch data from the image.
// 	void getMatrixSubset();
//
// 	// Generate a random number in the range [low high)
// 	double getRandomNumber(unsigned int low, unsigned int high);
//
// 	// Given two points on a cartesian plane, calculate the gradient.
// 	double calculateGradient(const Point& a, const Point& b);
//
// 	// Check if we hae exhausted the search space (all of direction[] == 1)
// 	bool exhaustedSearch();
//
// 	// Set up the stack with all initial patches.
// 	void loadStack();
//
// public:
// 	// Constructor.
// 	Patch(const Mat& img, const Point& pt, const Size& s);
//
// 	// Stringify.
// 	String toString();
//
// 	// Draw green box around the current patch location.
// 	void colourPatch(Mat& img);
//
// 	// Are there any more patches to check?
// 	bool hasMoves();
//
//
// 	bool reachedEnd(Direction d);
// 	void move(Direction d);
// 	void process(Mat& im);
// 	bool calculateNextMove();
// };
//
// // Constructor.
// // Parameters:
// // - Const ptr pointer to the underlying image.
// // - Starting point / origin which is the middle of the underlying image.
// // - A size object containing the length and width for the patch.
// Patch::Patch(const Mat& img, const Point& pt, const Size& s) : image(img) {
// 	// Simple assignments.
// 	origin = pt;
// 	width = s.width;
// 	height = s.height;
//
// 	// Generate a random number from [0 4).
// 	int randomDirection = (int) getRandomNumber(0, 4);
//
// 	// Remember our current direction.
// 	currentDirection = randomDirection;
//
// 	// Grab the actual patch data from underlying image.
// 	getMatrixSubset();
// }
//
// bool Patch::hasMoves() {
// 	return !moves.empty();
// }
//
// void Patch::loadStack() {
//
// }
//
// bool Patch::calculateNextMove() {
// 	// Stack is empty and we have searched everthing so we are done.
// 	if (moves.empty() && exhaustedSearch()) {
// 		return false;
// 	}
//
// 	// Stack is empty.
// 	// We are just starting or have finished one direction.
// 	// Load up the stack with all the starting patches.
// 	if (moves.empty()) {
// 		Point next;
//
// 		moves.push(next);
// 	}
// 	return true;
// }
//
// void Patch::isValidMove(Point& p) {
// 	if (p.x > 0 && p.x < image.size().height
// 		&& p.y > 0 && p.y < image.size().width) {
// 		return true;
// 	}
// 	return false;
// }
//
// void Patch::process(Mat& im) {
// 	// Adaptive threshold.
// 	Mat thresholdedPatch;
// 	adaptiveThreshold(data, thresholdedPatch, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3, -2);
//
// 	// Hough lines.
// 	vector<Vec4i> lines;
// 	HoughLinesP(thresholdedPatch, lines, 1, CV_PI/180, 10, 25, 5);
//
// 	// Report findings.
// 	if (lines.size() > 0) {
// 		cout << " Found " << lines.size() << " line(s)" << endl;
// 	}
//
// 	// Draw lines.
// 	// Move this to the guard above.
// 	// Instead of plotting each line. Just calculate the median gradient and plot
// 	// the line with THAT gradient on the main image.
// 	Mat copy;
// 	cvtColor(thresholdedPatch, copy, CV_GRAY2RGB);
// 	for(size_t i = 0; i < lines.size(); i++) {
// 		Point a = Point(origin.x + lines[i][0], origin.y + lines[i][1]);
// 		Point b = Point(origin.x + lines[i][2], origin.y + lines[i][3]);
// 		cout << a << endl;
// 		cout << b << endl;
//
// 		double g = calculateGradient(a, b);
// 		Point p(0,0), q(im.size().width,im.size().height);
// 		p.y = -(a.x - p.x) * g + a.y;
// 		q.y = -(b.x - q.x) * g + b.y;
//
// 		cout << "Line 1 gradient: " << g << endl;
// 		line(im, p, q, Scalar(100,255,0), 1, 8);
// 	}
//
// 	// Show patch (resized).
// 	Mat resized;
// 	resize(copy, resized, Size(300, 300));
// 	imshow("Patch", resized);
//
// 	// Push on to stack the next patch we want to look at.
// 	// Either we found a line in wich case the follow it.
// 	// Or we didn't and we just advance.
// 	// If we are at the end we don't push anything.
// }
//
// double Patch::calculateGradient(const Point& a, const Point& b) {
// 	return (((double) b.y - (double) a.y) / ((double) b.x - (double) a.x));
// }
//
// void Patch::getMatrixSubset() {
// 	image(Rect(origin.x, origin.y, width, height)).copyTo(data);
// }
//
// double Patch::getRandomNumber(unsigned int low, unsigned int high) {
// 	random_device rd;
// 	mt19937 gen(rd());
// 	uniform_real_distribution<double> dis(low, nextafter(high, DBL_MAX));
// 	return dis(gen);
// }
//
// bool Patch::exhaustedSearch() {
// 	for (auto d : directions) {
// 		if (d == 0) {
// 			return true;
// 		}
// 	}
// 	return false;
// }
//
// // Returns patch origin and size.
// String Patch::toString() {
// 	return "Patch = Origin(" + to_string(origin.x) + "," + to_string(origin.y)
// 		+ ") width: " + to_string(width)
// 		+ ", height: " + to_string(height)
// 		+ ". Direction: " + to_string(currentDirection)
// 		+ ". Number of moves:" + to_string(moves.size()) + "\n";
// }
//
// // Draws a green box on the original, for debugging purposes.
// void Patch::colourPatch(Mat& img) {
// 	Point pt1 = origin;
// 	unsigned int x2 = pt1.x + width;
// 	unsigned int y2 = pt1.y + height;
// 	Point pt2(x2, y2);
// 	rectangle(img, pt1, pt2, Scalar(100, 255, 0), 1, 8, 0);
// }
//
// bool Patch::reachedEnd(Direction d) {
// 	switch (d) {
// 		case 0:
// 			if (origin.y == 0) return true;
// 		case 1:
// 			if (origin.x == 0) return true;
// 		case 2:
// 			if (origin.y == image.size().height) return true;
// 		case 3:
// 			if (origin.x == image.size().width) return true;
// 	}
// 	return false;
// }
//
// // Need reduce patch size near edges of image, otherwise we get overlap.
// void Patch::move(Direction d) {
// 	switch (d) {
// 		case UP:
// 			if (origin.y < height) {
// 				origin.y = 0;
// 			} else {
// 				origin.y = origin.y - height;
// 			}
// 		break;
// 	}
// 	getMatrixSubset();
// }
//
// // Helper functions.
// // Return centre of image as a Point.
// Point getCentre(const Mat& image) {
// 	Size s = image.size();
// 	return Point(s.width / 2, s.height / 2);
// }
//
//
//
// int main(int argc, char **argv) {
// 	// Read input image.
// 	Mat original = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
// 	if(original.empty()) {
// 		cout <<  "Could not open or find the image." << endl;
// 		return -1;
// 	}
//
// 	// Don't need high-resolution photos for edge detection.
// 	Mat resized;
// 	resize(original, resized, Size(WIDTH, HEIGHT));
//
// 	// Colour copy for drawing coloured things.
// 	Mat copy;
// 	cvtColor(resized, copy, CV_GRAY2RGB);
//
// 	// Dilate to remove text features on the receipts.
// 	Mat dilated;
// 	Mat kernel = getStructuringElement(MORPH_RECT, Size(DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE));
// 	dilate(resized, dilated, kernel, Point(-1, -1), DILATION_ITERATIONS);
//
// 	// Get centre of image.
// 	Point centre = getCentre(dilated);
//
// 	// Set patch size.
// 	Size s = Size(PATCH_WIDTH, PATCH_HEIGHT);
//
// 	// Get a new patch.
// 	Patch p = Patch(dilated, centre, s);
//
// 	imshow("Display window", copy);
//
// 	do {
// 		cout << p.toString() << endl;
// 		p.colourPatch(copy);
// 		p.process(copy);
// 		imshow("Display window", copy);
// 		int k = waitKey(0);
// 		if(k == 27) {
// 			exit(0);
// 		}
// 	} while(p.calculateNextMove());
//
// 	return 0;
// }
