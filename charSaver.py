import cv2
import numpy as np

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import imutils
import keras
from keras.models import load_model

import glob
import random

import recognizePlate


def main():    
    all_images = glob.glob("savedPlates/*.jpg")
    if len(all_images)<=0:
        print('Images are not defined. Please, run plateSaver.py berfore this script.')      
    for img_name in all_images:
        print(img_name)
        img = cv2.imread(img_name)
        image = imutils.resize(img, height=300)
        thresh = recognizePlate.threshMaker(image)
        indexOfLastSlash = img_name.rfind('/')+1
        if img_name[indexOfLastSlash:indexOfLastSlash+3]=='001' or img_name[indexOfLastSlash:indexOfLastSlash+3]=='002':
            contours = recognizePlate.detectContoursInFirstTypePlate(thresh)
        elif img_name[indexOfLastSlash:indexOfLastSlash+3]=='003' or img_name[indexOfLastSlash:indexOfLastSlash+3]=='004':
            contours = recognizePlate.detectContoursInSecondTypePlate(thresh)
        else:
            continue
        if contours is None:
            print('NoContours')
            continue
        ind = 0
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.imwrite('savedChars/'+str(ind)+'_'+str(int(random.random()*10000))+'.jpg',thresh[y:y+h,x:x+w])        
            print('Contour saved',ind)
            ind+=1








    return
# end main

if __name__ == "__main__":
    main()



















