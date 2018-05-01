import cv2
import numpy as np

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
from darkflow.net.build import TFNet

import glob
import random

import sys

S = 5 #StepSize - Save image every 'S' frame
minHeight = 50 #min height of plate for saving

def main():

    
    options = {
        'model': 'carplate/carplate.cfg',
        'load': 'carplate/yolo-obj_22000.weights',
        'threshold': 0.3,
        'labels': 'carplate/obj.names',
        'gpu': 0.0,
        'gpuName':'/gpu:0'
    }

    tfnet = TFNet(options)
    # read the color image and covert to RGB
    if sys.argv[1] == 'video':
        all_videos = glob.glob("videosWithCars/*.mp4")
        if len(all_videos)<=0:
            print('Videos are not defined. Please, put your videos (mp4) into videosWithCars folder.')
        counter = 0
        for video in all_videos:
            cap = cv2.VideoCapture(video)
            print(video)
            try:
                while cap.isOpened():
                    ret, img = cap.read()  
                    if counter%S!=0:
                        counter+=1
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = tfnet.return_predict(img)
                    if len(result)==0:
                        continue
                    for res in result:
                        #d = int((res['bottomright']['x']-res['topleft']['x'])*0.1)
                        d=0
                        tl = (res['topleft']['x']-d, res['topleft']['y']-d)
                        br = (res['bottomright']['x']+d, res['bottomright']['y']+d)
                        label = res['label'] 
                        crop_img = img[res['topleft']['y']-d:res['bottomright']['y']+d, res['topleft']['x']-d:res['bottomright']['x']+d]
                        if counter%S==0 and crop_img.shape[0]>minHeight:
                            cv2.imwrite('savedPlates/'+label+'_'+str(counter)+'_'+str(int(random.random()*10000))+'.jpg',crop_img)
                            print('Frame is saved', counter)
                        counter = counter + 1
            except:
                print('Video reading Error, continued',Exception)
            print('All plates from Video files from folder videoWithCars are saved in the folder savedPlates')
    elif sys.argv[1]=='image':
        all_images = glob.glob("imagesWithCars/*.jpg")
        if len(all_images)<=0:
            print('Images are not defined. Please, put your images (.jpg) into imagesWithCars folder.')
        counter = 0
        for image_name in all_images:
            try:
                img = cv2.imread(image_name)
                print(image_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = tfnet.return_predict(img)
                if len(result)==0:
                    continue
                for res in result:
                    #d = int((res['bottomright']['x']-res['topleft']['x'])*0.1)
                    d=0
                    tl = (res['topleft']['x']-d, res['topleft']['y']-d)
                    br = (res['bottomright']['x']+d, res['bottomright']['y']+d)
                    label = res['label'] 
                    # add the box and label and display it
                    crop_img = img[res['topleft']['y']-d:res['bottomright']['y']+d, res['topleft']['x']-d:res['bottomright']['x']+d]
                    if crop_img.shape[0]>minHeight:
                        cv2.imwrite('savedPlates/'+label+str(counter)+'_'+str(int(random.random()*10000))+'.jpg',crop_img)
                        print('Image is saved', counter)
                    counter = counter + 1
            except:
                print('Image reading Error, continued',Exception)
        print('All plates from Image files from folder imagesWithCars are saved in the folder savedPlates')
    else:
        print('Argument error, Please Run with video or image argument. example: python3 plateSaver.py image')


###################################################################################################
if __name__ == "__main__":
    main()



















