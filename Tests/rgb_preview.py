import os
import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def showPipeline_RGB(_cam_rgb_pipeline: depthai.node.ColorCamera):
    
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
            
            
def showPipeline_Depth(_cam_pipeline: depthai.node.StereoDepth):
    
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
    

pipeline = depthai.Pipeline()
cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(1280, 720)
cam_rgb.setInterleaved(False)
showPipeline_RGB(cam_rgb)

# stereo_depth = pipeline.create(depthai.node.StereoDepth)
# stereo_threshold = 10 # 0-255 value, 0 is max confidence
# stereo_depth.initialConfig.setConfidenceThreshold(10)
# # Prioritize fill-rate, sets Confidence threshold to 245
# stereo_depth.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# # Prioritize accuracy, sets Confidence threshold to 200
# stereo_depth.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_ACCURACY)

# showPipeline_Depth(stereo_depth)

# detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
# # Set path of the blob (NN model). We will use blobconverter to convert&download the model
# # detection_nn.setBlobPath("/path/to/model.blob")
# detection_nn.setBlobPath(
#     os.path.join(
#         r"C:\Users\rocapabl\OneDrive - Otis Elevator\Escritorio\BlobModels",
#         "mobilenet-ssd_openvino_2021.4_6shave.blob"),
#     shaves=6)
# detection_nn.setConfidenceThreshold(0.5)

# cam_rgb.preview.link(detection_nn.input)

# xout_rgb = pipeline.create(depthai.node.XLinkOut)
# xout_rgb.setStreamName("rgb")
# cam_rgb.preview.link(xout_rgb.input)

# # xout_nn = pipeline.create(depthai.node.XLinkOut)
# # xout_nn.setStreamName("nn")
# # detection_nn.out.link(xout_nn.input)

# with depthai.Device(pipeline) as device:
#     q_rgb = device.getOutputQueue("rgb")
#     # q_nn = device.getOutputQueue("nn")
#     frame = None
#     detections = []
    
#     while True:
#         in_rgb = q_rgb.tryGet()
#         # in_nn = q_nn.tryGet()
        
#         if in_rgb is not None:
#             frame = in_rgb.getCvFrame()
            
#             # if in_nn is not None:
#             #     detections = in_nn.detections
            
#             if frame is not None:
#                 # for detection in detections:
#                 #     bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
#                 #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
#                 cv2.imshow("preview", frame)
                
#         if cv2.waitKey(1) == ord('q'):
#             break