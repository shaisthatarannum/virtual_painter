import cv2
import numpy as np
import time
import os
import handtrackingmodule as htm

#################

brushThickness = 15
eraserThickness = 80

#################


folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overLaylist=[]
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLaylist.append(image)
print(len(overLaylist))
header = overLaylist[0]
drawColor=(255,0,255)

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)

window_width = 800
window_height = 500

# Create the image window with the specified size
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", window_width, window_height)

# Create the canvas window with the specified size
cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Canvas", window_width, window_height)

# Create the inv window with the specified size
cv2.namedWindow("Inv", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Inv", window_width, window_height)


detector = htm.handDetector(detectionCon=0.85)
xp, yp= 0, 0
imgCanvas = np.zeros((720, 1200, 3),np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

#     # # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) !=0:
        
        #print(lmList)

        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

    #3

        fingers = detector.fingersUp(lmList)
        #print(fingers)

    #4
        if fingers[1] and fingers[2]:
            xp, yp= 0, 0
            print("Selection Mode")
            if y1<125:
                if 250<x1<450:
                    header=overLaylist[0]
                    drawColor=(255,0,255)
                elif 550<x1<750:
                    header=overLaylist[1]
                    drawColor=(255,0,0)
                elif 800<x1<950:
                    header=overLaylist[2]
                    drawColor=(0,255,0)
                elif 1050<x1<1200:
                    header=overLaylist[3]
                    drawColor=(0,0,0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25), drawColor,cv2.FILLED)

        #5
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(img, (xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1),drawColor,brushThickness)
            
            xp,yp = x1, y1

    imgCanvas = cv2.resize(imgCanvas, (img.shape[1], img.shape[0]))

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    
    imgInv = cv2.resize(imgInv, (imgCanvas.shape[1], imgCanvas.shape[0]))

    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
                          
    # Fix: Resize imgInv to match imgCanvas shape
    img[0:125,0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    
    cv2.imshow("Image",img)
    cv2.imshow("Canvas",imgCanvas)
    cv2.imshow("Inv",imgInv)
    cv2.waitKey(1)