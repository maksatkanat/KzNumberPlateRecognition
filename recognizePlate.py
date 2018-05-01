import cv2
import numpy as np
import imutils


#isPossibleChar - define countour wich can be real symbol in the plate, 
#returns True or False
def isPossibleChar(number,allPossibleChars):
    if number>=len(allPossibleChars):
        return False
    possibleChar = allPossibleChars[number]
    status = False
    scale = 0.15
    (x0, y0, w0, h0) = cv2.boundingRect(possibleChar)
    for c in allPossibleChars:
        (x, y, w, h) = cv2.boundingRect(c)
        if x0 == x and y0 == y:
            continue
        if (w0>w-w*scale and w0<w+w*scale) or (h0>h-h*scale and h0<h+h*scale) and y0>y-h*scale and y0<y+h*scale:
            return True
    return False
#end isPossibleChar


#threshMaker - returns adaptive Threshold from argument image
def threshMaker(image):
    border = 5
    image[0:border,:,:] = 255
    image[image.shape[0]-border:image.shape[0],:,:] = 255
    image[:,0:border,:] = 255
    image[:,image.shape[1]-border:image.shape[1],:] = 255

    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
    image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    adaptiveThresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)
    return adaptiveThresh
#end threshMaker

#recognizer - returns String, recognized text from threshold image, 
# using contours and tensorflow model.
def recognizer(thresh, contours, model):
    labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    res_text=''
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        img = thresh[y:y + h, x:x + w]
        res = cv2.resize(img, (28,28),interpolation = cv2.INTER_CUBIC)            
        im = asarray(res)
        im = im.reshape(1, 28, 28, 1)
        img=im.astype(np.float32)
        img=img/255
        pred = model.predict(img)*100
        
        if w<int(h*0.2):
            if (labels[np.argmax(pred)]=='1' or labels[np.argmax(pred)]=='I') and int(pred[0][np.argmax(pred)])>50:
                res_text=res_text + labels[np.argmax(pred)]

        elif int(pred[0][np.argmax(pred)])>50:
            res_text=res_text + labels[np.argmax(pred)]

    if len(res_text)<6:
        return None
    else:
        return res_text 
#end recognizer

#detectContoursInFirstTypePlate - returns contours of possible chars 
#from threshold image of 001 and 002 type plate numbers(one line text)
def detectContoursInFirstTypePlate(thresh):

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        if w >= thresh.shape[1]*0.01 and w <= thresh.shape[1]*0.15 and h >= 80 and h<290:
            digitCnts.append(c)


   
    
    i=0
    while i<len(digitCnts):
        if not isPossibleChar(i,digitCnts):
            digitCnts.pop(i)
            #print('popped 1 ')
        else:
            i=i+1
    i=0
    j=0
    # if second contour in first contour => delete second contour
    while i<len(digitCnts):
        try:
            (x1, y1, w1, h1) = cv2.boundingRect(digitCnts[i])
        except:
            break
        
        j=0
        while j<len(digitCnts):
            try:
                (x2, y2, w2, h2) = cv2.boundingRect(digitCnts[j])
            except:
                break

            if (x2>x1 and x2+w2<x1+w1 and y2>y1 and y2+h2<y1+h1):
                digitCnts.pop(j)
                #print('POP',j)
                if i>=1:
                    i=i-1
                    break
            j=j+1
        i=i+1
    
    

            
    i=0
    j=0
    
    for i in range (len(digitCnts)): 
        for j in range (0,len(digitCnts)-i-1):
            (x1, y1, w1, h1) = cv2.boundingRect(digitCnts[j])
            (x2, y2, w2, h2) = cv2.boundingRect(digitCnts[j+1])
            if x1 > x2:
                cnt = digitCnts[j]
                digitCnts[j] = digitCnts[j+1]
                digitCnts[j+1] = cnt

    

    '''if len(digitCnts)<8:
        return None
    else:
        digitCnts = digitCnts[len(digitCnts)-8:len(digitCnts)]'''


    return digitCnts
#end detectContoursInFirstTypePlate

#detectContoursInSecondTypePlate - returns contours of possible chars 
#from threshold image of 003 and 004 type plate numbers(two line text)
def detectContoursInSecondTypePlate(thresh):

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        if w >= thresh.shape[1]*0.08 and w <= thresh.shape[1]*0.3 and h >= thresh.shape[0]*0.2 and w<int(1.8*h) and h<thresh.shape[0]/2:
            digitCnts.append(c)


    i=0
    j=0
    
    # if second contour in first contour => delete second contour
    while i<len(digitCnts):
        try:
            (x1, y1, w1, h1) = cv2.boundingRect(digitCnts[i])
        except:
            break
        
        j=0
        while j<len(digitCnts):
            try:
                (x2, y2, w2, h2) = cv2.boundingRect(digitCnts[j])
            except:
                break

            if (x2>x1 and x2+w2<x1+w1 and y2>y1 and y2+h2<y1+h1):
                digitCnts.pop(j)
                #print('POP',j)
                if i>=1:
                    i=i-1
                    break
            j=j+1
        i=i+1
    i=0
    while i<len(digitCnts):
        if not isPossibleChar(i,digitCnts):
            digitCnts.pop(i)
        else:
            i=i+1

    #choosing top chars line and bottom chars line
    br = 0.5 #central line to find top chars line and bottom chars line
    topDigitCnts = []
    bottomDigitCnts = []
    x1=0
    y1=0
    h1=0
    w1=0
    for c in digitCnts:
        (x1, y1, w1, h1) = cv2.boundingRect(c)
        if y1+h1/2 < br*thresh.shape[0]:
            topDigitCnts.append(c)
            #print('Top:',x1,' ',y1)
        else:
            bottomDigitCnts.append(c)
            #print('Bottom:',x1,' ',y1)
    
    #Sort top chars
    i=0
    j=0
    for i in range (len(topDigitCnts)): 
        for j in range (0,len(topDigitCnts)-i-1):
            (x1, y1, w1, h1) = cv2.boundingRect(topDigitCnts[j])
            (x2, y2, w2, h2) = cv2.boundingRect(topDigitCnts[j+1])
            if x1 > x2:
                cnt = topDigitCnts[j]
                topDigitCnts[j] = topDigitCnts[j+1]
                topDigitCnts[j+1] = cnt
    #Sort bottom chars
    i=0
    j=0
    for i in range (len(bottomDigitCnts)):
        for j in range (0,len(bottomDigitCnts)-1-i):
            (x1, y1, w1, h1) = cv2.boundingRect(bottomDigitCnts[j])
            (x2, y2, w2, h2) = cv2.boundingRect(bottomDigitCnts[j+1])
            if x1 > x2:
                cnt = bottomDigitCnts[j]
                bottomDigitCnts[j] = bottomDigitCnts[j+1]
                bottomDigitCnts[j+1] = cnt

    '''if len(topDigitCnts)<3:
        return None
    else:
        topDigitCnts = topDigitCnts[len(topDigitCnts)-3:len(topDigitCnts)]

    if len(bottomDigitCnts)!=5:
        return None'''

    if len(bottomDigitCnts)==5:
        firstPart = bottomDigitCnts[0:2]
        secondPart = bottomDigitCnts[2:5]
        bottomDigitCnts[0:3] = secondPart
        bottomDigitCnts[3:5] = firstPart 
    if len(bottomDigitCnts)==4:
        firstPart = bottomDigitCnts[0:2]
        secondPart = bottomDigitCnts[2:4]
        bottomDigitCnts[0:3] = secondPart
        bottomDigitCnts[3:5] = firstPart 
    
    digitCnts=[]
    digitCnts = topDigitCnts
    for c in bottomDigitCnts:
        digitCnts.append(c)

    return digitCnts
#end detectContoursInSecondTypePlate

#drawContours - returns drawed image, using contours
def drawContours(img,contours):
    #print('Image Shape : ',img.shape)
    ind = 0
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        #print('w = ',w,'h = ',h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img
#end drawContours



#returs result text or None. Main method. 
# Arguments: 
#   img - car's image with plate; 
#   tfnet - ready Yolo model for plate detection;
#   model - tensorflow model for character recognition.
def startDetectAndRecognize(img,tfnet,model):

    result = tfnet.return_predict(img)
    if len(result)==0:
        return None
    #img.shape
    # pull out some info from the results 

    crop_img = 0
    max = 0


    for res in result:
        #d = int((res['bottomright']['x']-res['topleft']['x'])*0.1)
        d=0
        tl = (res['topleft']['x']-d, res['topleft']['y']-d)
        br = (res['bottomright']['x']+d, res['bottomright']['y']+d)
        if max<res['bottomright']['x']-res['topleft']['x']:
            crop_img = img[res['topleft']['y']-d:res['bottomright']['y']+d, res['topleft']['x']-d:res['bottomright']['x']+d]
            label = res['label'] 

    image = imutils.resize(crop_img, height=300)
    thresh = threshMaker(image)

    if label == '001' or label == '002':
        contours = detectContoursInFirstTypePlate(thresh)
    else:
        contours = detectContoursInSecondTypePlate(thresh)
    
    if contours is None:
        return None
    img = drawContours(image,contours)
    recognizedPlateText = recognizer(thresh, contours, model)
    return recognizedPlateText
#end startDetectAndRecognize