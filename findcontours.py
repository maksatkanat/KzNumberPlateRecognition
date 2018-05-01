from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

image = cv2.imread("rect_plate.jpg")
#cv2.imshow("Input", image)
image = imutils.resize(image, height=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#edged = cv2.Canny(blurred, 50, 200, 255)
#cv2.imshow('gr',gray)
#cv2.imshow('ed',blurred)

thresh = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if w >= 30 and h >= 30 and h<2*w and w<2*h:
        digitCnts.append(c)
t=0
print(len(digitCnts))
for i in range (len(digitCnts)):
    for j in range (len(digitCnts)-1):
        (x1, y1, w1, h1) = cv2.boundingRect(digitCnts[j])
        (x2, y2, w2, h2) = cv2.boundingRect(digitCnts[j+1])
        if x1 > x2:
            cnt = digitCnts[j]
            digitCnts[j] = digitCnts[j+1]
            digitCnts[j+1] = cnt



for c in digitCnts:
    (x, y, w, h) = cv2.boundingRect(c)
    roi = thresh[y:y + h, x:x + w]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.putText(image, str(t), (x, y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 1)
    cv2.imwrite('saved/'+str(t)+'.jpg',image[y:y+h,x:x+w])
    t=t+1
    
#cv2.imshow('tr',thresh)

cv2.imshow("Output", image)
cv2.waitKey(0)