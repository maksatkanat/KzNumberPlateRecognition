import cv2
import numpy as np

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from darkflow.net.build import TFNet

import imutils
import keras
from keras.models import load_model

import glob
import random

labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']


def main():
    
    options = {
        'model': 'carplate/carplate.cfg',
        'load': 'carplate/yolo-obj_22000.weights',
        'threshold': 0.3,
        'labels': 'carplate/obj.names',
        'gpu': 0.0,
        'gpuName':'/gpu:1'
    }


    cv2.namedWindow('main',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', 800,500)
    cv2.moveWindow("main", 0,500)


    tfnet = TFNet(options)
    model = load_model('my_model_syntezed_3004_with_inv.h5')
    cap = cv2.VideoCapture('videosWithCars/1.mp4')
    all_cars = glob.glob("cars/*.jpg")
    if len(all_cars)<=0:
        print('Images are not defined. Please, put your images (.jpg) into cars folder.')      
    for img_name in all_cars:
        img = cv2.imread(img_name)
        result = recognizePlate.startDetectAndRecognize(img,tfnet,model)

        #image = imutils.resize(img, height=300)
        #thresh = recognizePlate.threshMaker(image)
        #contours = recognizePlate.detectContoursInFirstTypePlate(thresh)
        #contours = detectContoursInSecondTypePlate(image,thresh)
        #if contours is None:
        #    continue
        #img = recognizePlate.drawContours(image,contours)
        #cv2.imshow('thresh',thresh)

        cv2.imshow('main',img)
        cv2.waitKey(0)








    return
# end main

if __name__ == "__main__":
    main()



















