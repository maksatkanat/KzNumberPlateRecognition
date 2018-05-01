import numpy as np
from pylab import *
from loadDataset import load_data_2d as load_data
from random import random
import cv2

#For testing dataset for training. Randomly shows data and print it's value.

(X_train, y_train, X_test, y_test) = load_data()

labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
t =0
while True:
    #t = int(random()*120000)
    if labels[np.argmax(y_train[t])] != ' ':
        im = np.reshape(X_train[t],(28,28))
        cv2.imshow('a',im)
        print(labels[np.argmax(y_train[t])])
        cv2.waitKey(0)
    t=t+1
cv2.destroyAllWindows()