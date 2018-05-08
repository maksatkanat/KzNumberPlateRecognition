import cv2
import os
import numpy
#6,9,4     J,I,Q,G
#img = cv2.imread('face.jpg')
s = ''
c = ''
'''os.mkdir('letters')
os.mkdir('letters-inverse')'''
#Create folders in letters
'''for i1 in range(ord('A'), ord('Z')+1):    
    os.mkdir('letters/'+str(chr(i1)))
    os.mkdir('letters-inverse/'+str(chr(i1)))
for j1 in range(0,10):
    os.mkdir('letters/'+str(j1))
    os.mkdir('letters-inverse/'+str(j1))'''
cols = 28
rows = 28
dataSize = 1000000
X =  numpy.zeros((dataSize,784),numpy.uint8)                   #166320
Y =  numpy.zeros((dataSize,1),numpy.uint8) 
counter = 0
y_label = 0
for i in range(ord('A'), ord('Z')+1):
    if i =='O':
        continue
    t = 0
    for angle in range(-5,6):
        for xpos in range(2,9):
            for ypos in range(26,30):
                for font in range(2,5):
                    for scale in range(9,14):
                        img = numpy.ones((cols,rows,1),numpy.uint8)*255
                        img_inv = numpy.zeros((cols,rows,1),numpy.uint8)
                        cv2.putText(img ,str(chr(i)), (xpos, ypos),cv2.FONT_HERSHEY_SIMPLEX, scale/10.0, (0, 0, 0), font) #FONT_HERSHEY_PLAIN
                        cv2.putText(img_inv ,str(chr(i)), (xpos, ypos),cv2.FONT_HERSHEY_SIMPLEX, scale/10.0, (255, 255, 255), font) #FONT_HERSHEY_PLAIN
                        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)  #-15 => +15
                        img = cv2.warpAffine(img,M,(cols,rows))
                        img_inv = cv2.warpAffine(img_inv,M,(cols,rows))
                        #cv2.imwrite('letters/'+str(chr(i))+'/'+str(t)+'.jpg',img)
                        #cv2.imwrite('letters-inverse/'+str(chr(i))+'/'+str(t)+'.jpg',img_inv)
                        t=t+1
                        
                        X[counter] = img_inv.ravel()
                        Y[counter] = y_label
                        #print(X[0])
                        print(str(counter) + ' ' + str(Y[counter]))
                        counter = counter+1
                        #cv2.imshow('inv', img_inv)
                        #cv2.waitKey(0)
    y_label=y_label+1
    s = s + str(chr(i))

for i in range(0, 10):
    t = 0
    for angle in range(-5,6):
        for xpos in range(2,9):
            for ypos in range(26,30):
                for font in range(2,5):
                    for scale in range(9,14):
                        img = numpy.ones((cols,rows,1),numpy.uint8)*255
                        img = numpy.ones((cols,rows,1),numpy.uint8)*255
                        cv2.putText(img ,str(i), (xpos, ypos),cv2.FONT_HERSHEY_SIMPLEX, scale/10.0, (0, 0, 0), font) #FONT_HERSHEY_PLAIN
                        cv2.putText(img_inv ,str(i), (xpos, ypos),cv2.FONT_HERSHEY_SIMPLEX, scale/10.0, (255, 255, 255), font) #FONT_HERSHEY_PLAIN
                        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)  #-15 => +15
                        img = cv2.warpAffine(img,M,(cols,rows))
                        img_inv = cv2.warpAffine(img_inv,M,(cols,rows))
                        #cv2.imwrite('letters/'+str(i)+'/'+str(t)+'.jpg',img)
                        #cv2.imwrite('letters-inverse/'+str(i)+'/'+str(t)+'.jpg',img_inv)
                        t=t+1
                        
                        X[counter] = img_inv.ravel()
                        Y[counter] = y_label
                        #print(X[0])
                        print(str(counter) + ' ' + str(Y[counter]))
                        counter = counter+1
                        #cv2.imshow('inv', img_inv)
                        #cv2.waitKey(0)
    y_label=y_label+1


numpy.savez('syntez-new.npz',X = X, Y = Y)

X=X[0:counter]
Y=Y[0:counter]
#for j in range(0,10):
#    c=c+str(j)
#s=s+c
#print(s)
#print(c)'''


'''cv2.destroyAllWindows()
cv2.putText(img ,s, (0, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5) #FONT_HERSHEY_PLAIN
cv2.imshow('img',img)
cv2.waitKey(0)'''
