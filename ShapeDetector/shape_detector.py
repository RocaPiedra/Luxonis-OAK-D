# import the necessary packages
import cv2
import depthai
import math
from aux_functions import VariableStringBuilder

class ShapeDetector:
	def __init__(self, min_area, max_area):
		self.min_area = min_area
		self.max_area = max_area
	def detect(self, contour, target_shape = None, only_closed_contours: bool = False):
		# initialize the shape name and approximate the contour
		closed_contour = True
		area = cv2.contourArea(contour) 
		perimeter = cv2.arcLength(contour, True)
		if only_closed_contours:
			closed_contour = self.verify_closed_contour(area, perimeter)
			if closed_contour is False: return None
			elif not self.verify_area(area, self.min_area, self.max_area): return None
		contour_approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True) # format?
		shape = ShapeDetector.return_shape_name(len(contour_approximation))
		if shape != "unidentified":
			if target_shape is None:
				return DetectedShapeClass(
					contour, contour_approximation, shape, None, is_target=True, is_closed=closed_contour)
			else:
				return DetectedShapeClass(
					contour, contour_approximation, shape, None, is_target=(shape==target_shape), is_closed=closed_contour)

	def get_contour(self, c, ratio, image, target_shape = None, paint_image: bool = True, only_closed_contours: bool = False):
		M = cv2.moments(c)		
		try:
			if  M["m00"] == 0:
				raise Exception("Zero Moments, contour is not correct")
			cX = int((M["m10"] / M["m00"]) * ratio)
			cY = int((M["m01"] / M["m00"]) * ratio)
			ShapeClass = self.detect(c, target_shape,only_closed_contours)
			if ShapeClass is not None:
				ShapeClass.correct_location((cX, cY))
				if paint_image:
					ShapeClass.paintShape(image, ratio)
				return ShapeClass
		except Exception as e:
			print("Error [get_contour]: " + str(e))
   	
	@staticmethod
	def verify_closed_contour(area, perimeter):
		# if the contour is open the area should be zero:
		if area > perimeter:
			return True
		else:
			return False
		
	@staticmethod
	def verify_min_area(area, min_area):
		# if the contour is open the area should be zero:
		if area > min_area:
			return True
		else:
			return False
	@staticmethod
	def verify_max_area(area, max_area):
		# if the contour is open the area should be zero:
		if area < max_area:
			return True
		else:
			return False
	@staticmethod
	def verify_area(area, min_area, max_area):
		# if the contour is open the area should be zero:
		if ShapeDetector.verify_min_area(area, min_area) and ShapeDetector.verify_max_area(area, max_area):
			return True
		else:
			return False

	@staticmethod
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
	def __init__(self, contour, poligon_approximation, shape_name, centroid_location, is_target = False, is_closed = True):
		self.contour = contour
		self.poligon_approximation = poligon_approximation
		self.shape_name = shape_name
		self.centroid_location = centroid_location
		self.is_target = is_target
		self.is_closed = is_closed
		self.ROI: depthai.Rect = None
		self.setROI()

	def correct_location(self, new_centroid_location):
		self.centroid_location = new_centroid_location
		self.setROI()
	
# translation from centroid to ROI compatible with SpatialLocationCalculator
	def setROI(self, _d: float = 100, _debug: bool = False):
		try:
			if self.centroid_location is None:
				if _debug: raise Exception("Centroid is not available")
				else: return None
			if  self.contour is not None:
				_d = self.getMinDistCentroid2Poligon(self.centroid_location, self.contour)
			self.ROI = self.generateROI(self.centroid_location, _d)
			return self.ROI
		except Exception as e:
			print("Error [setROI]: " + str(e))
			return None
	
	def getROI(self):
		if self.ROI is not None:
			return self.ROI
		else:
			# less accurate option, distance is not given
			return self.setROI()

# ROI is generated with the references of opencv images, 0,0 is top left corner and only positive values
	@staticmethod
	def generateROI(centroid: tuple, distance: float):
		if isinstance(centroid, tuple) and len(centroid) == 2:
			_x = centroid[0] #(cX, cY)
			_y = centroid[1] #(cX, cY)
			topLeft = depthai.Point2f((_x - distance*math.cos(45)), (_y - distance*math.sin(45)))
			bottomRight = depthai.Point2f((_x + distance*math.cos(45)), (_y + distance*math.sin(45)))
			return depthai.Rect(topLeft, bottomRight)
		else:
			raise ValueError("generateROI: Input must be a tuple defining the centroid location (cX, cY)")
			
	@staticmethod
	def getMinDistCentroid2Poligon(centroid: tuple, contour, min_size: int = 10):
		min_dist = 100
		for c in contour:
			try:
				new_dist = math.sqrt((centroid[0] - c[0][0])^2 + (centroid[1] - c[0][1])^2)
				if new_dist < min_dist and new_dist >= min_size:
					min_dist = new_dist
			except:
				pass
			return min_dist
				
	def paintShape(self, image, ratio):
		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the image
		c = self.contour.astype("float")
		c *= ratio
		c = c.astype("int")
		if self.is_target and self.is_closed:
			shape_color = (0, 128, 0) #green
			text_color = shape_color
		elif self.is_closed == False:
			return
			shape_color = (255, 255, 255) #white
			text_color = shape_color
		else:
			shape_color = (0, 0, 255) #red
			text_color = shape_color
		cv2.drawContours(image, [c], -1, shape_color, 2)
		cv2.putText(image, self.shape_name, self.centroid_location, cv2.FONT_HERSHEY_SIMPLEX,
			0.5, text_color, 2)

	def toString(self, _show: bool = False):		
		builder = VariableStringBuilder()
		if self.poligon_approximation is not None: builder.add_variable("Poligon", self.poligon_approximation)
		if self.shape_name is not None: builder.add_variable("Shape", self.shape_name)
		if self.centroid_location is not None: builder.add_variable("Centroid Location", self.centroid_location)
		if self.is_target is not None: builder.add_variable("Is Target", self.is_target)
		result = builder.build_string()
		if _show: (result)
		return result

