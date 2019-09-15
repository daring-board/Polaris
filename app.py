import cv2, json, os
import gc
import numpy as np
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_from_directory
import tensorflow as tf
from threading import Thread
import configparser
from keras.models import Sequential, load_model
from mtcnn.mtcnn import MTCNN
from fine_tuning import FineTuning

''' 設定ファイルの読み込み '''
config = configparser.ConfigParser()
config.read('./model/config.ini')

app = Flask(__name__)
UPLOAD_FOLDER = config['PATH']['upload']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PREDICT_FOLDER = config['PATH']['generate']
app.config['PREDICT_FOLDER'] = PREDICT_FOLDER

label_list = json.load(open(config['PATH']['category'], 'r'))
labels = {int(k): v for k, v in label_list.items()}
graph = tf.get_default_graph()
ft = FineTuning()
model = ft.createModel(label_list)
detector = MTCNN()

def load_model():
    global graph, model
    with graph.as_default():
        model.load_weights(config['PATH']['use_chkpnt'])

@app.route('/', methods = ["GET", "POST"])
def root():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        f = request.files['FILE']
        f_path = save_img(f)
        predict = json.loads(pred_org(f_path, f.filename).data.decode('utf-8'))
        print(predict)
        return render_template(
                    'index.html',
                    filepath=predict['path'],
                )

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

@app.route('/generate/predict/<filename>')
def generated_file(filename):
    return send_from_directory(app.config['PREDICT_FOLDER'], filename)

def save_img(f):
    stream = f.stream
    print(f.filename)
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    f_path = UPLOAD_FOLDER+'/'+f.filename
    cv2.imwrite(f_path, img)
    return f_path

def pred_org(f_path, filename):
    datas = []
    size = (int(config['PARAM']['width']), int(config['PARAM']['height']))
    img = cv2.imread(f_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(img)
    faces, bb = [], []
    for idx, item in enumerate(results):
        x1, y1, width, height = item['box']
        tmp = img[y1: y1+height, x1: x1+width,:]
        tmp = cv2.resize(tmp, (160, 160))
        tmp = tmp.astype(np.float32) / 255.0
        faces.append(tmp)
        bb.append(item['box'])
    faces = np.asarray(faces)

    with graph.as_default():
        predict = model.predict(faces)

    for idx, p in enumerate(predict):
        print(labels[np.argmax(p)])
        print(p)
        if max(p) < 0.2: continue
        x1, y1, width, height = bb[idx]
        cv2.rectangle(img, (x1,y1), (x1+width, y1+height), (255, 0, 0), 2)
        cv2.putText(img, labels[np.argmax(p)], (x1, y1+10),
            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 1, cv2.LINE_AA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    path = PREDICT_FOLDER+'/'+filename
    cv2.imwrite(path, img)

    return jsonify({
            'status': 'OK',
            'path': path
        })

if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
