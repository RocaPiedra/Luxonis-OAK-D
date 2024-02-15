import cv2
import numpy as np 
from matplotlib import pyplot as plt 
from calibrator_class import *
from aux_functions import *
from TestImages.image_searcher import *
import time

def ConcatenateImages(images, target_res):
    images3D = []
    if len(images) == 2:
        target_res = (target_res[0],(int)(target_res[1]/2))
    for image in images:
        if image.ndim == 2:
            images3D.append(cv2.merge((image, image, image)))
        else:
            images3D.append(image)            
    combined_image = np.hstack(images3D)
    return (cv2.resize(combined_image, target_res))

def Preproc4ShapeDetection(img, show_images = True):
    assert img is not None
    
    kernel = np.ones((5,5),np.uint8)
 
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
    imgCanny = cv2.Canny(img,150,200)
    imgDilation = cv2.dilate(imgCanny,kernel,iterations=1)
    imgEroded = cv2.erode(imgDilation,kernel,iterations=1)
    
    if show_images:
        plt.subplot(2,3,1),plt.imshow(img)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,2),plt.imshow(imgGray, cmap='gray')
        plt.title('Gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,3),plt.imshow(imgBlur, cmap='gray')
        plt.title('Gaussian Blur'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,4),plt.imshow(imgCanny, cmap='gray')
        plt.title('Canny'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,5),plt.imshow(imgDilation, cmap='gray')
        plt.title('Dilation'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,6),plt.imshow(imgEroded, cmap='gray')
        plt.title('Erosion'), plt.xticks([]), plt.yticks([])
        plt.show()
        
def warpPerspective(img):
    width,height = 250,350
    pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    warped = cv2.warpPerspective(img,matrix,(width,height))
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(warped)
    plt.title('Gray'), plt.xticks([]), plt.yticks([])
    return 
    
def readWebcam(deviceCapture = 0, frameWidth = 640, frameHeight = 480):
    
    cap = cv2.VideoCapture(deviceCapture)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 150)
    arr_invalid = []
    while True:
        success, img = cap.read()
        if not success:
            arr_invalid.append(deviceCapture)
            deviceCapture += 1
            cap = cv2.VideoCapture(deviceCapture)
        else:
            cv2.imshow("Result", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    
            
def createWindowTrackbar(_Calibrator):    
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars",640,240)
    cv2.createTrackbar("CannyLowThreshold","TrackBars",
                       _Calibrator.low_threshold,
                       _Calibrator.low_max,
                       _Calibrator.LowChange)    
    cv2.createTrackbar("CannyHighThreshold","TrackBars",
                       _Calibrator.high_threshold,
                       _Calibrator.high_max,
                       _Calibrator.HighChange)
    
def CannyListProcessor(images, change_time = 10):
    
    cannyCalibrator = Calibrator(200,300,400,0,500,200)
    createWindowTrackbar(cannyCalibrator)
    
    for image in images:
        img = imageScaler(image,20)
        # warped = warpPerspective(img)
        # cv2.imshow('shapes', warped) 

        start = time.time()

        while time.time() - start < change_time:
            
            imgCanny = cv2.Canny(img,
                                cannyCalibrator.low_threshold,
                                cannyCalibrator.high_threshold)
            cv2.imshow("Canny", imgCanny)
            cv2.waitKey(1)
            
        start = time.time()
        
def CannyPipeline():
    
    cannyCalibrator = Calibrator(200,300,400,0,500,200)
    createWindowTrackbar(cannyCalibrator)

def main():
    
    images = image_finder()
    CannyListProcessor(images)
        
    
if __name__ == "__main__":
    main()