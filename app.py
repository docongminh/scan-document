import os
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import scan
import numpy as np
import cv2

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('FLASK_UPLOAD_FOLDER')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/scan', methods=['POST'])
def scan_document():
    # check if the post request has the file part
    if 'image_file' not in request.files:
        return jsonify(error='Missing image file')

    image_file = request.files['image_file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if image_file.filename == '':
        return jsonify(error='File is empty')

    if image_file and not allowed_file(image_file.filename):
        return jsonify(error='Not support file type')

    img_str = image_file.read()
    array = np.fromstring(img_str, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    contour, warped = scan.crop_then_transform_document(image)
    text = scan.extract_text_from_image(warped)
    return jsonify(text=text, contour=contour)
