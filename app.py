# https://stackoverflow.com/questions/46785507/python-flask-display-image-on-a-html-page
# https://jdhao.github.io/2020/04/12/build_webapi_with_flask_s2/
# python3 app.py
# source env/bin/activate

from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from PIL import Image
import cv2
import numpy as np
import os
from Bayesian_Crowd_Counting.preprocess_dataset import preprocess
from Bayesian_Crowd_Counting.test import test_image
import scipy.io as sio

import sys

app = Flask(__name__)


@app.route('/')
def index():
    # return "Hellow World"
    return render_template('index.html')


@app.route('/im_size', methods=["POST"])
def process_image():
    file = request.files['image']

    # Save file
    file.save('static/processed/test/img.jpg')

    # Read the image via file.stream
    img = Image.open(file.stream)

    # Save .mat
    vect = np.array(img)
    # sio.savemat('static/processed/test/img_ann.mat', {'vect': vect})

    # Save .npy
    np.save('static/processed/test/img.npy', vect)

    # return send_file(img, mimetype='image/gif')
    # return jsonify({'msg': 'success', 'size': [img.width, img.height]})

    print("@method_preprocess()")
    # preprocess()

    print("@method_test_image()")
    out_name, out_temp_minu, out_count, out_sum = test_image()

    print("@method_return()")

    # DOWNLOAD
    return send_from_directory('static/final_photo', 'img.jpg', as_attachment=True)

    # VISUALIZE | Need to clear cache
    # return render_template("output2.html", user_image='static/people_photo/img.jpg')

    # PARAMS
    # return jsonify({
    #     'msg': 'success',
    #     'size': [img.width, img.height],
    #     'name': out_name,
    #     'temp_minu': out_temp_minu,
    #     'count': out_count,
    #     'sum': out_sum
    # })


@app.route('/get')
def get_image():
    return render_template("output2.html", user_image='static/final_photo/img.jpg')


# PEOPLE_FOLDER = os.path.join('static', 'people_photo')
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
# @app.route('/')
# @app.route('/index')
# def show_index():
#     full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
#     print(full_filename)
#     return render_template("output2.html", user_image=full_filename)


if __name__ == "__main__":
    app.run(debug=False)
