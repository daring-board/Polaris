import cv2, json, os
import numpy as np
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_from_directory
import tensorflow as tf
from threading import Thread
from keras.models import Sequential, load_model

from pred_model import FineTuning
import grade_cam as gc

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
HEATMAP_FOLDER = './generate/heatmap'
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER

label_list = list(json.load(open('./model/category.json', 'r')).keys())
graph = tf.get_default_graph()
ft = FineTuning(len(label_list), 'VGG16')
model = ft.createNetwork()

def load_model():
    global graph, model
    with graph.as_default():
        model.load_weights('./model/checkpoints/weights.17-0.03-0.99-0.02-0.99.hdf5')

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
            lines += '%s: %.3f<br/>'%(item[0], item[1])
        org_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        return render_template(
                    'index.html',
                    filepath=org_path,
                    heatmapath=heatmap(f.filename),
                    context=lines
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
    global graph
    datas = []
    img = cv2.imread(f_path)
    img = cv2.resize(img, (128, 128))
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
    global graph, model
    f_path = UPLOAD_FOLDER+'/'+f_name
    heatmap = HEATMAP_FOLDER+'/heatmap_'+f_name
    with graph.as_default():
        guided_model = gc.build_guided_model()
        gradcam, gb, guided_gradcam = gc.compute_saliency(model, guided_model, layer_name='block5_conv3',
                                         img_path=f_path, cls=-1, visualize=False, save=False)
        cv2.imwrite(heatmap, gc.deprocess_image(guided_gradcam[0]))

    return heatmap

if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run(host='0.0.0.0',port=5000)
