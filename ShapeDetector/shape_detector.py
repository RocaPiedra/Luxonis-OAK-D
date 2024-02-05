# import the necessary packages
import cv2
class ShapeDetector:
	def __init__(self):
		pass
	def detect(self, contour, target_shape = None):
		# initialize the shape name and approximate the contour
		
		perimeter = cv2.arcLength(contour, True)
		contour_approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
		shape = ShapeDetector.return_shape_name(len(contour_approximation))
		if shape != "unidentified":
			if target_shape is None:
				return DetectedShapeClass(contour_approximation, shape, None, is_target=True)
			else:
				if shape == target_shape:
					return DetectedShapeClass(contour_approximation, shape, None, is_target=True)
				else:
					return DetectedShapeClass(contour_approximation, shape, None, is_target=False)


	def get_contour(self, c, ratio, image, target_shape = None):
		M = cv2.moments(c)
		try:
			if  M["m00"] == 0:
				raise Exception("Zero Moments, contour is not correct")
			cX = int((M["m10"] / M["m00"]) * ratio)
			cY = int((M["m01"] / M["m00"]) * ratio)
			ShapeClass = self.detect(c, target_shape)
			ShapeClass.correct_location((cX, cY))
			if ShapeClass is not None:
			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the name of the shape on the image
				c = c.astype("float")
				c *= ratio
				c = c.astype("int")
				if ShapeClass.is_target:
					color = (0, 128, 0)
				else:
					color = (0, 0, 255)
				cv2.drawContours(image, [c], -1, color, 2)
				cv2.putText(image, ShapeClass.shape_name, ShapeClass.centroid_location, cv2.FONT_HERSHEY_SIMPLEX,
					0.5, (255, 255, 255), 2)
				return ShapeClass
		except Exception as e:
			print(str(e))
   
	def return_shape_name(vertex_count: int):
		match vertex_count:
			case vertex_count if vertex_count < 3:
				shape = "unidentified"
			case vertex_count if vertex_count == 3:
				shape = "triangle"
			case vertex_count if vertex_count == 4:
				(x, y, w, h) = cv2.boundingRect(vertex_count)
				ar = w / float(h)
				shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
			case vertex_count if vertex_count == 5:
				shape = "pentagon"
			case vertex_count if vertex_count == 6:
				shape = "hexagon"
			# case vertex_count if vertex_count == 7:
			# 	shape = "heptagon"
			# case vertex_count if vertex_count == 8:
			# 	shape = "octagon"
			# case vertex_count if vertex_count == 9:
			# 	shape = "nonagon"
			# case vertex_count if vertex_count == 10:
			# 	shape = "decagon"
			# case vertex_count if vertex_count == 11:
			# 	shape = "hendecagon"
			# case vertex_count if vertex_count == 12:
			# 	shape = "dodecagon"
			case _:
				shape = "circle"
    
		return shape

class DetectedShapeClass:    
	def __init__(self, poligon_approximation, shape_name, centroid_location, is_target = False):
		self.poligon_aproximation = poligon_approximation
		self.shape_name = shape_name
		self.centroid_location = centroid_location
		self.is_target = is_target

	def correct_location(self, new_centroid_location):
		self.centroid_location = new_centroid_location
  