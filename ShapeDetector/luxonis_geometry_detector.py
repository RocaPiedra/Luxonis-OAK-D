import os
import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs
import cv_functions
import calibrator_class
import imutils
from shape_detector import ShapeDetector
from dataclasses import dataclass

@dataclass
class Resolution:
    width: int = 1280
    height: int = 720

# default values for canny algorithm with good results
@dataclass
class CannyCalibration:
    low: int = 25
    high: int = 241
    


class HoleDetector():        
    
    def __init__(self, rgb_res: Resolution = Resolution(1280,720), depth_res: Resolution = Resolution(1280,720)):
        self.pipeline = depthai.Pipeline()
        # prepare rgb stream for edge detection     
        self.cam_rgb = self.pipeline.create(depthai.node.ColorCamera)
        
        self.prepareRGBPipeline(rgb_res)
        
        self.shape_detector = ShapeDetector()
        
        self.kernel = np.ones((5,5),np.uint8)

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def showPipeline_RGB(pipeline: depthai.Pipeline, _cam_rgb_pipeline: depthai.node.ColorCamera):        
        xout_rgb = pipeline.create(depthai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        _cam_rgb_pipeline.preview.link(xout_rgb.input)
        with depthai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue("rgb")
            frame = None        
            while True:
                in_rgb = q_rgb.tryGet()            
                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()                
                    if frame is not None:
                        cv2.imshow("preview", frame)
                        
                if cv2.waitKey(1) == ord('q'):
                    break
                
    def showPipeline_Depth(pipeline: depthai.Pipeline, _cam_pipeline: depthai.node.StereoDepth):
        
        xout_rgb = pipeline.create(depthai.node.XLinkOut)
        xout_rgb.setStreamName("depth")
        _cam_pipeline.preview.link(xout_rgb.input)
        with depthai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue("depth")
            frame = None        
            while True:
                in_rgb = q_rgb.tryGet()            
                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()                
                    if frame is not None:
                        cv2.imshow("depth", frame)
                        
                if cv2.waitKey(1) == ord('q'):
                    break
                
    def prepareRGBPipeline(self, res: Resolution):    
        
        self.cam_rgb.setPreviewSize(res.width, res.height)
        self.cam_rgb.setInterleaved(False)
        self.xout_rgb = self.pipeline.create(depthai.node.XLinkOut)
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)
        return

    def prepareSpatialLocationCalculator(self, xOutDepth, xoutSpatialData, xinSpatialCalcConfig):
            # Define sources and outputs
        monoLeft = self.pipeline.create(depthai.node.MonoCamera)
        monoRight = self.pipeline.create(depthai.node.MonoCamera)
        stereo = self.pipeline.create(depthai.node.StereoDepth)
        spatialLocationCalculator = self.pipeline.create(depthai.node.SpatialLocationCalculator)

        xOutDepth = self.pipeline.create(depthai.node.XLinkOut)
        xoutSpatialData = self.pipeline.create(depthai.node.XLinkOut)
        xinSpatialCalcConfig = self.pipeline.create(depthai.node.XLinkIn)

        xOutDepth.setStreamName("depth")
        xoutSpatialData.setStreamName("spatialData")
        xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

        # Properties
        monoLeft.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setCamera("left")
        monoRight.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setCamera("right")

        stereo.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        # Config
        topLeft = depthai.Point2f(0.4, 0.4)
        bottomRight = depthai.Point2f(0.6, 0.6)

        config = depthai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 100
        config.depthThresholds.upperThreshold = 10000
        calculationAlgorithm = depthai.SpatialLocationCalculatorAlgorithm.MEDIAN
        config.roi = depthai.Rect(topLeft, bottomRight)

        spatialLocationCalculator.inputConfig.setWaitForMessage(False)
        spatialLocationCalculator.initialConfig.addROI(config)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        spatialLocationCalculator.passthroughDepth.link(xOutDepth.input)
        stereo.depth.link(spatialLocationCalculator.inputDepth)

        spatialLocationCalculator.out.link(xoutSpatialData.input)
        xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    def getShapes(self, preprocessed_img, rgb_img, target_shape = "hexagon", ratio = 1):    
        if preprocessed_img is not None:
            cnts = cv2.findContours(preprocessed_img.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)	
            cnts = imutils.grab_contours(cnts)	# check utility
            shapeList = []
            # loop over the contours
            for c in cnts:
                if c is not None:
                    shapeList.append(self.shape_detector.get_contour(c, ratio, rgb_img, target_shape))           
            
            return rgb_img

    def Preproc4ShapeDetection(self, img, canny_low_thresh, canny_high_thresh):
        assert img is not None
            
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
        imgCanny = cv2.Canny(imgBlur,canny_low_thresh,canny_high_thresh)
        imgDilation = cv2.dilate(imgCanny,self.kernel,iterations=1)
        imgEroded = cv2.erode(imgDilation,self.kernel,iterations=1)
        
        return imgEroded

def main():
    
    cannyCalibrator = calibrator_class.Calibrator(CannyCalibration.low,CannyCalibration.high,400,0,500,200)
    cv_functions.createWindowTrackbar(cannyCalibrator)
    detector = HoleDetector(Resolution(1280,720))
    
    with depthai.Device(detector.pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        frame = None        
        while True:
            in_rgb = q_rgb.tryGet()            
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()                
                if frame is not None:
                    # cv2.imshow("preview", frame)
                    # imgCanny = cv2.Canny(frame,
                    #                     cannyCalibrator.low_threshold,
                    #                     cannyCalibrator.high_threshold)
                    imgPreproc = detector.Preproc4ShapeDetection(frame, 
                                                    cannyCalibrator.low_threshold,
                                                    cannyCalibrator.high_threshold)
                    cv2.imshow("Canny", imgPreproc)
                    imgShapes = detector.getShapes(imgPreproc, frame)
                    cv2.imshow("Shapes", imgShapes)
                    
            if cv2.waitKey(1) == ord('q'):
                break
            
if __name__ == '__main__':
    main()