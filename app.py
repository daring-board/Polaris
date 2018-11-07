import cv2, json, os
import gc
import numpy as np
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_from_directory
import tensorflow as tf
from threading import Thread
import configparser
from keras.models import Sequential, load_model

from pred_model import FineTuning
import grade_cam as gcam

''' 設定ファイルの読み込み '''
config = configparser.ConfigParser()
config.read('./model/config.ini')

app = Flask(__name__)
UPLOAD_FOLDER = config['PATH']['upload']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
HEATMAP_FOLDER = config['PATH']['heatmap']
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER

label_list = list(json.load(open(config['PATH']['category'], 'r')).keys())
name_dict = json.load(open(config['PATH']['name'], 'r', encoding="utf-8"))
graph = tf.get_default_graph()
ft = FineTuning(config, len(label_list))
model = ft.createNetwork()

def load_model():
    global graph, model
    with graph.as_default():
        model.load_weights(config['PATH']['use_chkpnt'])

@app.route('/', methods = ["GET", "POST"])
def root():
    c_names = [name_dict[key] for key in name_dict.keys()]
    if request.method == 'GET':
        return render_template('index.html', categorys=c_names)
    elif request.method == "POST":
        f = request.files['FILE']
        f_path = save_img(f)
        predict = pred_org(f_path).data.decode('utf-8')
        predict = json.loads(predict)['data']
        predict = sorted(predict.items(), key=lambda x:-x[1])
        lines = ''
        for item in predict[:3]:
            lines += '%s: %.3f<br/>'%(name_dict[item[0]], item[1])
        org_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        return render_template(
                    'index.html',
                    filepath=org_path,
                    heatmapath=heatmap(f.filename),
                    context=lines,
                    categorys=c_names
                )

@app.route('/ar')
def ar():
    return render_template('ar.html')

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

@app.route('/generate/heatmap/<filename>')
def heatmap_file(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename)

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
    size = (int(config['PARAM']['width']), int(config['PARAM']['height']))
    img = cv2.imread(f_path)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    datas.append(img)
    datas = np.asarray(datas)
    with graph.as_default():
        pred_class = model.predict(datas)
    ret = {label_list[idx]: float(pred_class[0][idx]) for idx in range(len(label_list))}

    return jsonify({
            'status': 'OK',
            'data': ret
        })

def heatmap(f_name):
    f_path = UPLOAD_FOLDER+'/'+f_name
    heatmap = HEATMAP_FOLDER+'/heatmap_'+f_name
    with graph.as_default():
        guided_model = gcam.build_guided_model(config)
        gradcam, gb, guided_gradcam = gcam.compute_saliency(model, guided_model, layer_name='block5_conv3',
                                         img_path=f_path, cls=-1, visualize=False, save=False)
        # cv2.imwrite(heatmap, gcam.deprocess_image(guided_gradcam[0]))
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + gcam.load_image(f_path, preprocess=False)) / 2
        cv2.imwrite(heatmap, np.uint8(jetcam))
    del guided_model
    gc.collect()
    return heatmap

if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run(host='0.0.0.0',port=5000)
