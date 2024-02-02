import cv2

def imageScalerDimensions(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return dim

def imageScaler(img, scale_percent):
    dim = imageScalerDimensions(img, scale_percent)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    