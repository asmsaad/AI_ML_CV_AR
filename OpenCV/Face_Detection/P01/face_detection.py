# Face tracker using OpenCV and Arduino
# by Shubham Santosh


import cv2 as cv
import numpy as np

# ('./../../../opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
# face_cascade= cv.CascadeClassifier('haarcascade_frontalface_default.xml') # From https://github.com/opencv/opencv/tree/master/data
# face_cascade= cv.CascadeClassifier('./../../../opencv-master/data/haarcascades/haarcascade_frontalface_default.xml') # From https://github.com/opencv/opencv/tree/master/data
face_cascade= cv.CascadeClassifier('./../../../opencv-master/data/haarcascades/haarcascade_profileface.xml') # From https://github.com/opencv/opencv/tree/master/data
def check01() :
    # cap = cv.VideoCapture("video01.mp4")
    cap = cv.VideoCapture("trackingVideo.mp4")
    cap = cv.VideoCapture(0)
    while(1):
        # Take each frame
        _, frame = cap.read()
   
   
   
   
        frame=cv.flip(frame,1)  #mirror the image
        #print(frame.shape)
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces= face_cascade.detectMultiScale(gray,1.1,6)  #detect the face
        for x,y,w,h in faces:
            #sending coordinates to Arduino
            string='X{0:d}Y{1:d}'.format((x+w//2),(y+h//2))
            # print(string)
            # ArduinoSerial.write(string.encode('utf-8'))
            #plot the center of the face
            cv.circle(frame,(x+w//2,y+h//2),2,(0,255,0),2)
            #plot the roi
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        #plot the squared region in the center of the screen
        cv.rectangle(frame,(640//2-30,480//2-30),
                (640//2+30,480//2+30),
                (255,255,255),3)
   
   
   
   
   
        cv.imshow('frame',frame)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()
    
check01()