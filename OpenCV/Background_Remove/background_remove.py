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



remove_background("messi5.jpg","messi5_new.jpg")
# remove_background("robot_new.png","robot_new_2.png")

