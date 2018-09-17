import cv2
import numpy as np
from flask import Flask, jsonify, request

from pred_model import FineTuning

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'

model = None
label_list = [
    'FATE',
    'HA',
    'HAGAREN',
    'MADOMAGI',
    'SAO',
    'TOARU'
]

def loadModel():
    global model
    model_file_name = "funiture_cnn.h5"
    ft = FineTuning(6, 'VGG16')
    model = ft.createNetwork()
    model.load_weights('checkpoints/weights.49-0.33-0.89-0.23-0.93.hdf5')

@app.route('/uploads', methods = ["POST"])
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

@app.route("/predict")
def predict():
    name = request.args.get('name')
    f = UPLOAD_FOLDER + '/' + name + '.jpg'
    print(f)
    return pred_org(f)

def pred_org(f_path):
    datas = []
    img = cv2.imread(f_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    datas.append(img)
    datas = np.asarray(datas)
    pred_class = model.predict(datas)
    ret = {label_list[idx]: float(pred_class[0][idx]) for idx in range(len(label_list))}

    return jsonify({
            'status': 'OK',
            'data': ret
        })


if __name__ == "__main__":
    loadModel()
    print(" * Flask starting server...")
    app.run()
