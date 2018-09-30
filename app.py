import cv2, json, os
import numpy as np
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_from_directory
import tensorflow as tf
from threading import Thread
from keras.models import Sequential, load_model

from pred_model import FineTuning

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

label_list = list(json.load(open('./model/category.json', 'r')).keys())
graph = tf.get_default_graph()
ft = FineTuning(len(label_list), 'VGG16')

@app.route('/', methods = ["GET", "POST"])
def root():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        f = request.files['FILE']
        f_path = save_img(f)
        predict = pred_org(f_path).data.decode('utf-8')
        predict = json.loads(predict)['data']
        predict = sorted(predict.items(), key=lambda x:-x[1])
        lines = ''
        for item in predict[:3]:
            lines += '%s: %.3f<BR>'%(item[0], item[1])
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        return render_template('index.html', filepath=path, context=lines)

@app.route('/predict', methods = ["POST"])
def uploads():
    if request.method == "POST":
        f = request.files['FILE']
        stream = f.stream
        print(f.filename)
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        f_path = UPLOAD_FOLDER+'/'+f.filename
        cv2.imwrite(f_path, img)
        return pred_org(f_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def save_img(f):
    stream = f.stream
    print(f.filename)
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    f_path = UPLOAD_FOLDER+'/'+f.filename
    cv2.imwrite(f_path, img)
    return f_path

def pred_org(f_path):
    datas = []
    img = cv2.imread(f_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    datas.append(img)
    datas = np.asarray(datas)
    with graph.as_default():
        model = ft.createNetwork()
        model.load_weights('./model/checkpoints/weights.07-0.36-0.89-0.17-0.95.hdf5')
        pred_class = model.predict(datas)
    ret = {label_list[idx]: float(pred_class[0][idx]) for idx in range(len(label_list))}

    return jsonify({
            'status': 'OK',
            'data': ret
        })

if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run()
