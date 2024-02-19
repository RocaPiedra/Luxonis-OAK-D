import os
import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs
import cv_functions
import calibrator_class
from typing import List
import depth_functions
from hole_detector import *
import random


def main():    
    RGBRes = Resolution(1920,1080)
    DepthRes = Resolution(640,400)
    VisRes = (1920,1000)
    # 1280×720, 1280×800, 640×400, 640×480, 1920×1200
    cannyCalibrator = calibrator_class.Calibrator(CannyCalibration.low,CannyCalibration.high,400,0,500,200)
    cv_functions.createWindowTrackbar(cannyCalibrator, "Detections")
    detector = HoleDetector(depthai.ColorCameraProperties.SensorResolution.THE_1080_P,RGBRes,
                            depthai.MonoCameraProperties.SensorResolution.THE_400_P)
    
    filter_open_contours = True
    
    with depthai.Device(detector.pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=False) # set the max size to stop processing old images
        depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False) # set the max size to stop processing old images
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=1, blocking=False)
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
        device.setIrLaserDotProjectorBrightness(750) # in mA, 0..1200 - over 750 saturates (Docs)
        device.setIrFloodLightBrightness(0)
        
        frame = None

        while True:
            shape_acquired = False
            depth_acquired = False
            # remove detected shapes each iteration
            detector.shapeList.clear()
            # get at the same time RGB and Depth -> Improve sync (To Do)
            in_rgb = q_rgb.get() # non-blocking
            in_depth = depthQueue.get() # non-blocking
            # in_spatial_calc = spatialCalcQueue.tryget
            if in_depth is not None:
                depthFrame = in_depth.getFrame()
                depth_downscaled = depthFrame[::4]
                if np.all(depth_downscaled == 0):
                    min_depth = 0  # Set a default minimum depth value when all elements are zero
                else:
                    min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
                max_depth = np.percentile(depth_downscaled, 99)
                depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
                depth_acquired = True
                # cv2.imshow("depth", depthFrameColor)
                
                    
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()                
                if frame is not None:
                    imgPreproc = detector.Preproc4ShapeDetection(frame, 
                                                    cannyCalibrator.low_threshold,
                                                    cannyCalibrator.high_threshold)
                    canny_scaled = cv2.resize(imgPreproc, VisRes)
                    # cv2.imshow("Canny", canny_scaled)
                    imgShapes = detector.getShapes(imgPreproc, frame, only_closed_contours=filter_open_contours)
                    detection_scaled = cv2.resize(imgShapes, VisRes)
                    # cv2.imshow("Shapes", detection_scaled)
                    concatenated = cv_functions.stack_images([canny_scaled, detection_scaled], 2, VisRes)
                    cv2.imshow("Detections", concatenated)
                    for shape in detector.shapeList:
                        if shape is not None:                            
                            shape.toString()
                            if shape.is_target:       
                                shape.getROI()
                                if shape.ROI is not None:
                                    # test de prueba                
                                    # test_ROI = depthai.Rect(depthai.Point2f(0.0, 0.0), depthai.Point2f(0.1, 0.1))
                                    # spatialCalcConfigInQueue.send(detector.setROIConfigTest(test_ROI))
                                    spatialCalcConfigInQueue.send(detector.setROIConfig(shape.ROI, RGBRes, DepthRes))
                                    shape_acquired = True                    
                                    if shape_acquired and depth_acquired:
                                        spatialData = spatialCalcQueue.get().getSpatialLocations()
                                        for depthData in spatialData:
                                            depth_functions.paintSpatialData(depthData, depthFrameColor)
                                            concatenated = cv_functions.stack_images([canny_scaled, detection_scaled, depthFrameColor], 2, VisRes)
                                            cv2.imshow("Detections", concatenated)
                                            # cv2.imshow("Detections", depthFrameColor)
            
              
            if cv2.waitKey(1) == ord('q'):
                break
            
if __name__ == '__main__':
    main()