import cv2
import numpy as np 
from matplotlib import pyplot as plt 
from calibrator_class import *
from aux_functions import *
from TestImages.image_searcher import *
import time

def pad_image(image, target_width):
    height, width = image.shape[:2]
    padding = np.zeros((height, target_width - width, 3), dtype=np.uint8)
    padded_image = np.hstack((image, padding))
    return padded_image

def stack_images(images, max_horizontal, full_window_resolution):
    num_images = len(images)
    rows = int(np.ceil(num_images / max_horizontal))
    stacked_images = []    
    prepared_images = []

    target_width = full_window_resolution[0] // max_horizontal
    target_height = full_window_resolution[1] // max_horizontal
    target_resolution = (target_width, target_height)

    while len(images) < max_horizontal * rows:
        images.append(np.zeros(target_resolution, np.uint8))

    for image in images:
        image = cv2.resize(image, target_resolution)
        if image.ndim == 2:
            prepared_images.append(cv2.merge((image, image, image)))
        else:
            prepared_images.append(image)

    for i in range(rows):
        row_images = prepared_images[i * max_horizontal: (i + 1) * max_horizontal]
        padded_row_images = [pad_image(img, target_width) for img in row_images]
        stacked_row = np.hstack(padded_row_images)
        stacked_images.append(stacked_row)

    final_image = np.vstack(stacked_images)

    final_image = cv2.resize(final_image, full_window_resolution)

    return final_image

def ConcatenateImages(horizontal_images: [], vertical_images: [], visualization_res):
    horiz_images = []
    vert_images = []
    frame_res = (visualization_res[0]/len(vertical_images),(int)(visualization_res[1]/len(horizontal_images)))
    if len(horizontal_images) == 2:
        visualization_res = (visualization_res[0],(int)(visualization_res[1]/2))
    for image in horizontal_images:
        if image.ndim == 2:
            horiz_images.append(cv2.merge((image, image, image)))
        else:
            horiz_images.append(image) 
    for image in vertical_images:
        if image.ndim == 2:
            vert_images.append(cv2.merge((image, image, image)))
        else:
            vert_images.append(image)

    result = None
    if len(horiz_images > 0): 
        result = np.hstack(horiz_images)      
    if len(vert_images > 0): 
        if result is None:
            combined_image = np.vstack(vert_images)
    return (cv2.resize(combined_image, visualization_res))

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
            
def createWindowTrackbar(_Calibrator: Calibrator, window_name: str = "TrackBars"):    
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name,640,240)
    cv2.createTrackbar("CannyLowThreshold",window_name,
                       _Calibrator.low_threshold,
                       _Calibrator.low_max,
                       _Calibrator.LowChange)    
    cv2.createTrackbar("CannyHighThreshold",window_name,
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