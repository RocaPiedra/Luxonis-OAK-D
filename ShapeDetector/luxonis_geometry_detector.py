import os
import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs
import cv_functions
import calibrator_class
import imutils
from shape_detector import ShapeDetector

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
            
def prepareRGBPipeline(pipeline, cam_rgb):    
    
    cam_rgb.setPreviewSize(1280, 720)
    cam_rgb.setInterleaved(False)
    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    return 

def getShapes(preprocessed_img, target_shape = None):    
    if preprocessed_img is not None:
        cnts = cv2.findContours(preprocessed_img.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)	
        cnts = imutils.grab_contours(cnts)	# check utility
        sd = ShapeDetector()
        # loop over the contours
        for c in cnts:
            if c is not None:
                sd.get_contour(c, 1, preprocessed_img, target_shape)
        
        return preprocessed_img

def main():
    
    cannyCalibrator = calibrator_class.Calibrator(200,300,400,0,500,200)
    cv_functions.createWindowTrackbar(cannyCalibrator)
    pipeline = depthai.Pipeline()
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    prepareRGBPipeline(pipeline, cam_rgb)
    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        frame = None        
        while True:
            in_rgb = q_rgb.tryGet()            
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()                
                if frame is not None:
                    cv2.imshow("preview", frame)
                    imgCanny = cv2.Canny(frame,
                                        cannyCalibrator.low_threshold,
                                        cannyCalibrator.high_threshold)
                    cv2.imshow("Canny", imgCanny)
                    imgShapes = getShapes(imgCanny, "hexagon")
                    cv2.imshow("Shapes", imgShapes)
                    
            if cv2.waitKey(1) == ord('q'):
                break
            
if __name__ == '__main__':
    main()