



#Face tracker using OpenCV and Arduino
#by Shubham Santosh


import cv2 as cv
import numpy as np


face_cascade= cv.CascadeClassifier('haarcascade_frontalface_default.xml') # From https://github.com/opencv/opencv/tree/master/data
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
    
# check01()




import cv2
import os
from rembg import remove
from PIL import Image
from werkzeug.utils import secure_filename
# from flask import Flask,request,render_template

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','webp'])

if 'static' not in os.listdir('.'):
    os.mkdir('static')

if 'uploads' not in os.listdir('static/'):
    os.mkdir('static/uploads')

# app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def remove_background(input_path,output_path):
    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)


# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/remback',methods=['POST'])
# def remback():
#     file = request.files['file']
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         rembg_img_name = filename.split('.')[0]+"_rembg.png"
#         remove_background(UPLOAD_FOLDER+'/'+filename,UPLOAD_FOLDER+'/'+rembg_img_name)
#         return render_template('home.html',org_img_name=filename,rembg_img_name=rembg_img_name)


# if __name__ == '__main__':
#     app.run(debug=True)