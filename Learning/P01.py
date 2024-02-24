
# READ IMAGE
# import cv2 as cv
# import sys
# img = cv.imread(cv.samples.findFile("starry_night.jpg"))
# if img is None:
#     sys.exit("Could not read the image.")
# cv.imshow("Display window", img)
# k = cv.waitKey(0)
# if k == ord("s"):
#     cv.imwrite("starry_night.png", img)



# READ VIDEO
# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # Our operations on the frame come here
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break
# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()




# DRAWING
# import numpy as np
# import cv2 as cv
# # Create a black image
# img = np.zeros((512,512,3), np.uint8)
# # Draw a diagonal blue line with thickness of 5 px
# cv.line(img,(0,0),(511,511),(255,0,0),5)

# font = cv.FONT_HERSHEY_SIMPLEX
# cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

# cv.rectangle(img,(384,0),(510,128),(0,255,0),3)

# cv.circle(img,(447,63), 63, (0,0,255), -1)

# cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)


# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
# cv.polylines(img,[pts],True,(0,255,255))

# cv.imshow("Display window", img)
# k = cv.waitKey(0)








#Mouse as a Paint-Brush
# list all available mouse events available,
# import cv2 as cv
# events = [i for i in dir(cv) if 'EVENT' in i]
# print( events )

# Type 01
# import numpy as np
# import cv2 as cv
# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     if event == cv.EVENT_LBUTTONDBLCLK:
#         cv.circle(img,(x,y),100,(255,0,0),-1)
# # Create a black image, a window and bind the function to window
# img = np.zeros((512,512,3), np.uint8)
# cv.namedWindow('image')
# cv.setMouseCallback('image',draw_circle)
# while(1):
#     cv.imshow('image',img)
#     if cv.waitKey(1) & 0xFF == 27:
#         break


# Type02
# import numpy as np
# import cv2 as cv
# drawing = False # true if mouse is pressed
# mode = False # if True, draw rectangle. Press 'm' to toggle to curve
# ix,iy = -1,-1
# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode
#     if event == cv.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y
#     elif event == cv.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#             else:
#                 cv.circle(img,(x,y),5,(0,0,255),-1)
#     elif event == cv.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         else:
#             cv.circle(img,(x,y),5,(0,0,255),-1)


# img = np.zeros((512,512,3), np.uint8)
# cv.namedWindow('image')
# cv.setMouseCallback('image',draw_circle)
# while(1):
#     cv.imshow('image',img)
#     k = cv.waitKey(1) & 0xFF
#     if k == ord('m'):
#         mode = not mode
#     elif k == 27:
#         break
# cv.destroyAllWindows()



# Trackbar as the Color Palette
# import numpy as np
# import cv2 as cv
# def nothing(x):
#     pass
# # Create a black image, a window
# img = np.zeros((300,512,3), np.uint8)
# cv.namedWindow('image')
# # create trackbars for color change
# cv.createTrackbar('R','image',0,255,nothing)
# cv.createTrackbar('G','image',0,255,nothing)
# cv.createTrackbar('B','image',0,255,nothing)
# # create switch for ON/OFF functionality
# switch = '0 : OFF \n1 : ON'
# cv.createTrackbar(switch, 'image',0,1,nothing)
# while(1):
#     cv.imshow('image',img)
#     k = cv.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     # get current positions of four trackbars
#     r = cv.getTrackbarPos('R','image')
#     g = cv.getTrackbarPos('G','image')
#     b = cv.getTrackbarPos('B','image')
#     s = cv.getTrackbarPos(switch,'image')
#     if s == 0:
#         img[:] = 0
#     else:
#         img[:] = [b,g,r]
# cv.destroyAllWindows()








# #todo Basic Operations on Images
# import numpy as np
# import cv2 as cv
# img = cv.imread('messi5.jpg')

# # Image Shape
# print( img.shape )
# #Size
# print( img.size )
# # Data Type
# print( img.dtype )

# # ball = img[280:340, 330:390]
# # img[273:333, 100:160] = ball

# #? Warning
# #? cv.split() is a costly operation (in terms of time). So use it only if necessary. Otherwise go for Numpy indexing.
# b,g,r = cv.split(img)
# img = cv.merge((b,g,r))

# # cv.imshow("B", b)
# # cv.imshow("G", g)
# # cv.imshow("R", r)

# #? Suppose you want to set all the red pixels to zero - you do not need to split the channels first. Numpy indexing is faster:
# img[:,:,2] = 0

# #? Making Borders for Images (Padding)
# #* See https://docs.opencv.org/4.9.0/d3/df2/tutorial_py_basic_ops.html

# cv.imshow("Display window", img)
# k = cv.waitKey(0)
# if k == ord("q"):
#     exit()







# #Todo Arithmetic Operations on Images
# #todo Learn several arithmetic operations on images, like addition, subtraction, bitwise operations, and etc.
# import cv2 as cv
# #? Image Blending
# # Must be two image in same dimention
# img1 = cv.imread('opencvLogo.png')
# img2 = cv.imread('robot.png')
# dst = cv.addWeighted(img1,0.7,img2,0.3,0)

# cv.imshow('dst',dst)
# cv.waitKey(0)
# cv.destroyAllWindows()




# import cv2 as cv
# # Bitwise Operations
# # This includes the bitwise AND, OR, NOT, and XOR operations
# # Load two images
# img1 = cv.imread('messi5.jpg')
# img2 = cv.imread('opencvLogo.png')
# assert img1 is not None, "file could not be read, check with os.path.exists()"
# assert img2 is not None, "file could not be read, check with os.path.exists()"
# # I want to put logo on top-left corner, So I create a ROI
# rows,cols,channels = img2.shape
# roi = img1[0:rows, 0:cols]
# # Now create a mask of logo and create its inverse mask also
# img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
# mask_inv = cv.bitwise_not(mask)
# # Now black-out the area of logo in ROI
# img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# # Take only region of logo from logo image.
# img2_fg = cv.bitwise_and(img2,img2,mask = mask)
# # Put logo in ROI and modify the main image
# dst = cv.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst

# cv.imshow('res',img1)
# cv.waitKey(0)
# cv.destroyAllWindows()






# # todo Performance Measurement and Improvement Techniques
# import cv2 as cv
# e1 = cv.getTickCount()
# # your code execution
# e2 = cv.getTickCount()
# time = (e2 - e1)/ cv.getTickFrequency()

# print(e1,e2,time)

# img1 = cv.imread('messi5.jpg')
# assert img1 is not None, "file could not be read, check with os.path.exists()"
# e1 = cv.getTickCount()
# for i in range(5,49,2):
#     img1 = cv.medianBlur(img1,i)
# e2 = cv.getTickCount()
# t = (e2 - e1)/cv.getTickFrequency()
# print( t )
# # Result I    got is 0.521107655 seconds
# # Result Saad got is 0.2970789   seconds

# # ?Default Optimization in OpenCV
# # cv.useOptimized()
# #  %timeit res = cv.medianBlur(img,49)\

# #* visit https://wiki.python.org/moin/PythonSpeed/PerformanceTips











# #todo Image Processing in OpenCV
# #todo Image Processing in OpenCV
# #todo Image Processing in OpenCV

# #todo Changing Colorspaces
#     # In this tutorial, you will learn how to convert images from one color-space to another, like BGR ↔ Gray, BGR ↔ HSV, etc.
#     # In addition to that, we will create an application to extract a colored object in a video
#     # You will learn the following functions: cv.cvtColor(), cv.inRange(), etc.

# # There are more than 150 color-space conversion methods available in OpenCV. But we will look into only two, which are most widely used ones: BGR ↔ Gray and BGR ↔ HSV.
# # For color conversion, we use the function cv.cvtColor(input_image, flag) where flag determines the type of conversion.
# # For BGR → Gray conversion, we use the flag cv.COLOR_BGR2GRAY. Similarly for BGR → HSV, we use the flag cv.COLOR_BGR2HSV. To get other flags, just run following commands in your Python terminal:

# import cv2 as cv
# flags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print( flags )

# #? Note
# # For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255]. Different software use different scales. So if you are comparing OpenCV values with them, you need to normalize these ranges.

# #* Object Tracking

import cv2 as cv
import numpy as np

def check01() :
    # cap = cv.VideoCapture("video01.mp4")
    cap = cv.VideoCapture("trackingVideo.mp4")
    cap = cv.VideoCapture(0)
    while(1):
        # Take each frame
        _, frame = cap.read()
        # Convert BGR to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # define range of blue color in HSV
        #? For video01.mp4
        # lower_blue = np.array([110,50,50])
        # upper_blue = np.array([130,255,255])
        ## Threshold the HSV image to get only blue colors
        # mask = cv.inRange(hsv, lower_blue, upper_blue)
        #? For trackingVideo.mp4
        lower_green = np.array([50 ,50, 50])
        upper_green = np.array([70, 255, 255])
        ## Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, lower_green, upper_green)
        # Bitwise-AND mask and original image
        res = cv.bitwise_and(frame,frame, mask= mask)
        cv.imshow('frame',frame)
        cv.imshow('mask',mask)
        cv.imshow('res',res)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()
    
check01()

# #? How to find HSV values to track?
# # This is a common question found in stackoverflow.com. It is very simple and you can use the same function, cv.cvtColor(). 
# # Instead of passing an image, you just pass the BGR values you want. For example, to find the HSV value of Green, try the following commands in a Python terminal:
# # 196,163,208
# # R    B   G
# # 163,208,196
# # B    G   R
# # green = np.uint8([[[0,255,0 ]]])
# green = np.uint8([[[0,255,0]]])
# hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
# print( hsv_green )
# # [[[ 60 255 255]]]
# # Now you take [H-10, 100,100] and [H+10, 255, 255] as the lower bound and upper bound respectively. 
# # Apart from this method, you can use any image editing tools like GIMP or any online converters to 
# # find these values, but don't forget to adjust the HSV ranges.







# #?Geometric Transformations of Images
# #* Learn to apply different geometric transformations to images, like translation, rotation, affine transformation etc.
# #todo Scaling
# import numpy as np
# import cv2 as cv
# img = cv.imread('messi5.jpg')
# assert img is not None, "file could not be read, check with os.path.exists()"
# res = cv.resize(img,None,fx=4, fy=4, interpolation = cv.INTER_CUBIC)
# #OR
# height, width = img.shape[:2]
# res = cv.resize(img,(4*width, 4*height), interpolation = cv.INTER_CUBIC)

# cv.imshow('img',res)
# cv.waitKey(0)
# cv.destroyAllWindows()


# #todo Translation
# import numpy as np
# import cv2 as cv
# img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# rows,cols = img.shape
# Tx = 50
# Ty = 100
# M = np.float32([[1,0,Tx],
#                 [0,1,Ty ]])
# dst = cv.warpAffine(img,M,(cols,rows))
# cv.imshow('img',dst)
# cv.waitKey(0)
# cv.destroyAllWindows()



#todo Rotation
import numpy as np
import cv2 as cv
img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
rows,cols = img.shape
# cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))

dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()




#? Additional Resources
# "Computer Vision: Algorithms and Applications", Richard Szeliski