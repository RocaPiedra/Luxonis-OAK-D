# import the necessary packages
from shape_detector import ShapeDetector
import argparse
import imutils
import cv2
import os


def preprocess(input_image):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)[1]

def main():
	cwd = os.getcwd()
	# construct the argument parse and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to the input image")
	# args = vars(ap.parse_args())

	# if args["image"] is not None:
	#     image_path = args["image"]
	# else:
	# image_path = r'\ShapeDetector\TestImages\hexagons.png'
	image_path = r'\ShapeDetector\TestImages\colored_shapes.png'
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	image = cv2.imread(cwd + image_path)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
	resized = imutils.resize(image, width=300)
	cv2.imshow("resized", resized)
	cv2.waitKey(0)
	ratio = image.shape[0] / float(resized.shape[0])
	thresh = preprocess(resized)
	cv2.imshow("thresh", thresh)
	cv2.waitKey(0)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	try:
		cv2.imshow("cnts", cnts)
		cv2.waitKey(0)
	except:
		pass
	cnts = imutils.grab_contours(cnts)
	try:
		cv2.imshow("cnts", cnts)
		cv2.waitKey(0)
	except:
		pass
	sd = ShapeDetector()

	# loop over the contours
	for c in cnts:
		sd.get_contour(c, ratio, image)
		# show the output image
		cv2.imshow("Image", image)
		cv2.waitKey(0)
  
if __name__ == "__main__":
    main()