import os
import glob
import cv2

def image_finder():
    images = []
    dir = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(dir)
    for file in files:
        if file.endswith(".png"):
            images.append(cv2.imread(os.path.join(dir,file)))
    
    return images

if __name__ == "__main__":
    images = image_finder()
    if len(images) == 0:
        print ("No images found")