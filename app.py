# 参考: https://www.python.ambitious-engineer.com/archives/1630
# 参考: https://note.com/kamakiriphysics/n/n2aec5611af2a
# 参考:  https://qiita.com/Gen6/items/2979b84797c702c858b1

import os
import random
from glob import glob
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, g, flash
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.utils.data
from torchvision.utils import make_grid, save_image

from generate_images_2 import generate_image # モジュールのインポート（オリジナルのコードを変更）

app = Flask(__name__)

SAVE_DIR = "images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

@app.route('/images/<path:filepath>')
def send_js(filepath):
    return send_from_directory(SAVE_DIR, filepath)

@app.route("/", methods=["GET","POST"])
def upload_photo():
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":
        kind = request.form.get('kind')
        if kind:
            if kind == "real":
                img_list = glob('./images/real/*.png')
                image_path_1 = random.sample(img_list, 1)
                return render_template("index.html", image_path_1=image_path_1[0])
            elif kind == "fake":
                model_path = './model/'
                image_path_2 = "./images/fake/fake_" + datetime.now().strftime("%Y%m%d%H%M%S_") + ".png"
                #generate_image(option, model_path, image_path_2)
                generate_image(model_path, image_path_2)
                return render_template("index.html", image_path_2=image_path_2)      
        else: # エラー処理
            return render_template("index.html", err_message="Please select a type.")

if __name__ == '__main__':
    app.run(debug=True,  host='0.0.0.0', port=5554) # ポートの変更
