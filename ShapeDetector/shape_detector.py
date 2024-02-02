# import the necessary packages
import cv2
class ShapeDetector:
	def __init__(self):
		pass
	def detect(self, contour, target_shape = None):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		perimeter = cv2.arcLength(contour, True)
		contour_approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        # if the shape is a triangle, it will have 3 vertices
		if len(contour_approximation) == 3:
			shape = "triangle"
		elif len(contour_approximation) == 4:
			(x, y, w, h) = cv2.boundingRect(contour_approximation)
			ar = w / float(h)
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
		elif len(contour_approximation) == 5:
			shape = "pentagon"
		elif len(contour_approximation) == 6:
			shape = "hexagon"
		else:
			shape = "circle"
		if target_shape is None:
			return shape
		else:
			if shape == target_shape:
				return shape
			else:
				return None


	def get_contour(self, c, ratio, image, target_shape = None):
		M = cv2.moments(c)
		try:
			if  M["m00"] == 0:
				raise Exception("Zero Moments, contour is not correct")
			cX = int((M["m10"] / M["m00"]) * ratio)
			cY = int((M["m01"] / M["m00"]) * ratio)
			shape = self.detect(c, target_shape)
			if shape is not None:
			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the name of the shape on the image
				c = c.astype("float")
				c *= ratio
				c = c.astype("int")
				cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
				cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, (255, 255, 255), 2)			
		except Exception as e:
			print(str(e))


class DetectedShape:    
	def __init__(self, poligon_approximation, shape_name):
		self.poligon_aproximation = poligon_approximation
		self.shape_name = shape_name