import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import imutils
from shape_detector import ShapeDetector
from dataclasses import dataclass
from typing import List

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
    
    def __init__(self, 
                 rgb_res:depthai.ColorCameraProperties.SensorResolution = depthai.ColorCameraProperties.SensorResolution.THE_12_MP, 
                 rgb_preview_res: Resolution = Resolution(1280,720), 
                 depth_res: depthai.MonoCameraProperties.SensorResolution = depthai.MonoCameraProperties.SensorResolution.THE_400_P):
        self.pipeline = depthai.Pipeline()
        # prepare rgb stream for edge detection     
        self.cam_rgb = self.pipeline.create(depthai.node.ColorCamera)
        
        self.monoLeft = self.pipeline.create(depthai.node.MonoCamera)
        self.monoRight = self.pipeline.create(depthai.node.MonoCamera)
        self.stereo = self.pipeline.create(depthai.node.StereoDepth)
        self.spatialLocationCalculator = self.pipeline.create(depthai.node.SpatialLocationCalculator)
        
        self.prepareRGBPipeline(rgb_res, rgb_preview_res)
        self.prepareSpatialLocationCalculator(depth_res)
        
        self.shape_detector = ShapeDetector()
        
        self.kernel = np.ones((5,5),np.uint8)
        
        self.shapeList: List[ShapeDetector.DetectedShapeClass] = []
        self.TargetShapeList: List[ShapeDetector.DetectedShapeClass] = []

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
                
    def prepareRGBPipeline(self, res: depthai.ColorCameraProperties.SensorResolution, preview_res: Resolution):    
        
        self.cam_rgb.setResolution(res)
        self.cam_rgb.setPreviewSize(preview_res.width, preview_res.height)
        self.cam_rgb.setInterleaved(False)
        self.xout_rgb = self.pipeline.create(depthai.node.XLinkOut)
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)
        return

    def prepareSpatialLocationCalculator(self, 
                                         monoRes:depthai.MonoCameraProperties.SensorResolution = depthai.MonoCameraProperties.SensorResolution.THE_400_P,
                                         stereoProfile:depthai.node.StereoDepth.PresetMode = depthai.node.StereoDepth.PresetMode.HIGH_ACCURACY):
            # Define sources and outputs
        self.xOutDepth = self.pipeline.create(depthai.node.XLinkOut)
        self.xoutSpatialData = self.pipeline.create(depthai.node.XLinkOut)
        self.xinSpatialCalcConfig = self.pipeline.create(depthai.node.XLinkIn)

        self.xOutDepth.setStreamName("depth")
        self.xoutSpatialData.setStreamName("spatialData")
        self.xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

        # Properties
        self.monoLeft.setResolution(monoRes)
        self.monoLeft.setCamera("left")
        self.monoRight.setResolution(monoRes)
        self.monoRight.setCamera("right")

        self.stereo.setDefaultProfilePreset(stereoProfile)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(True)

        # Config        
        self.initROIConfig()
        
        # Linking
        self.monoLeft.out.link(self.stereo.left)
        self.monoRight.out.link(self.stereo.right)

        self.spatialLocationCalculator.passthroughDepth.link(self.xOutDepth.input)
        self.stereo.depth.link(self.spatialLocationCalculator.inputDepth)
        # self.stereo.depth.link(self.xOutDepth.input)

        self.spatialLocationCalculator.out.link(self.xoutSpatialData.input)
        self.xinSpatialCalcConfig.out.link(self.spatialLocationCalculator.inputConfig)
        
    def initROIConfig(self, new_ROI: depthai.Rect = depthai.Rect(depthai.Point2f(0.4, 0.4), depthai.Point2f(0.6, 0.6)), 
                     algorithm: depthai.SpatialLocationCalculatorAlgorithm = depthai.SpatialLocationCalculatorAlgorithm.MEDIAN):
        self.config = depthai.SpatialLocationCalculatorConfigData()
        self.config.depthThresholds.lowerThreshold = 100
        self.config.depthThresholds.upperThreshold = 10000
        
        # init ROI to none to avoid measuring distance all the time 
        self.config.roi = new_ROI # depthai.Rect(topLeft, bottomRight)
        self.config.calculationAlgorithm = algorithm
        self.spatialLocationCalculator.inputConfig.setWaitForMessage(False)
        self.spatialLocationCalculator.initialConfig.addROI(self.config)
    
    def setROIConfigTest(self, ROI: depthai.Rect,
                     algorithm: depthai.SpatialLocationCalculatorAlgorithm = depthai.SpatialLocationCalculatorAlgorithm.MEDIAN):
        self.config.roi = ROI
        self.config.calculationAlgorithm = algorithm
        cfg = depthai.SpatialLocationCalculatorConfig()
        cfg.addROI(self.config)
        
        return cfg
    
    def setROIConfig(self, new_ROI: depthai.Rect, resRGB: Resolution, resDepth: Resolution,
                     algorithm: depthai.SpatialLocationCalculatorAlgorithm = depthai.SpatialLocationCalculatorAlgorithm.MEDIAN):
        
        self.config.roi = self.scaleROI(new_ROI, resRGB, resDepth)
        self.config.calculationAlgorithm = algorithm
        cfg = depthai.SpatialLocationCalculatorConfig()
        cfg.addROI(self.config)
        
        return cfg
    
    @staticmethod
    def scaleROI(new_ROI: depthai.Rect, resRGB: Resolution, resDepth: Resolution):
        return HoleDetector.OpenCV2DepthAI(new_ROI, resRGB, resDepth)
    
    @staticmethod
    def OpenCV2DepthAI(cv2_rect: depthai.Rect, cv2_res: Resolution, dAI_res: Resolution):
        width_ratio = dAI_res.width/cv2_res.width
        height_ratio = dAI_res.height/cv2_res.height
        
        adapted_x = cv2_rect.x * width_ratio # multiply by width resolution ratio
        adapted_y = cv2_rect.y * height_ratio # multiply by height resolution ratio
        new_ROI_width = cv2_rect.width * width_ratio
        new_ROI_height = cv2_rect.height * height_ratio
        if new_ROI_height == 0:
            new_ROI_height = 10
        if new_ROI_width == 0:
            new_ROI_width = 10
               
        newTopLeft = depthai.Point2f((adapted_x - new_ROI_width/2)/dAI_res.width, (adapted_y - new_ROI_height/2)/dAI_res.height)
        newBottomRight = depthai.Point2f((adapted_x + new_ROI_width/2)/dAI_res.width, (adapted_y + new_ROI_height/2)/dAI_res.height)
        return  depthai.Rect(newTopLeft, newBottomRight)
    
    def getShapes(self, preprocessed_img, rgb_img, target_shape = "hexagon", ratio = 1):    
        if preprocessed_img is not None:
            cnts = cv2.findContours(preprocessed_img.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)	
            cnts = imutils.grab_contours(cnts)	# check utility
            self.shapeList = []
            # loop over the contours
            for c in cnts:
                if c is not None:
                    # shapeList stores all contours in a frame, the information is held in a list of DetectedShapeClass
                    self.shapeList.append(self.shape_detector.get_contour(c, ratio, rgb_img, target_shape))           
            
            return rgb_img # return the original image with the overlayed detection

    def Preproc4ShapeDetection(self, img, canny_low_thresh, canny_high_thresh):
        assert img is not None
            
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
        imgCanny = cv2.Canny(imgBlur,canny_low_thresh,canny_high_thresh)
        imgDilation = cv2.dilate(imgCanny,self.kernel,iterations=1)
        imgEroded = cv2.erode(imgDilation,self.kernel,iterations=1)
        
        return imgEroded