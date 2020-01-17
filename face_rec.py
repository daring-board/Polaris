import os, cv2, json, pickle
import configparser
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from fine_tuning import FineTuning

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('./model/config.ini')
    label_dict = pickle.load(open('./model/polaris_labels.pkl','rb'))
    labels = {v: k for k, v in label_dict.items()}

    detector = MTCNN()
    ft = FineTuning()
    model = ft.createModel(label_dict)
    model_path = './model/polaris_facenet_model.h5'
    model.load_weights(model_path)

    test_path = './test/'
    dir_list = os.listdir(test_path)
    f_list =[]
    for d in dir_list:
        path = test_path+d
        if os.path.isfile(path): continue
        for f in os.listdir(path):
            f_list.append(path+'/'+f)
    results = []
    for f in f_list:
        im = cv2.imread(f)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
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

        predict = model.predict(faces)
        print(f)
        for idx, p in enumerate(predict):
            print(labels[np.argmax(p)])
            print(p)
            if max(p) < 0.35: continue
            x1, y1, width, height = bb[idx]
            cv2.rectangle(img, (x1,y1), (x1+width, y1+height), (255, 0, 0), 2)
            cv2.putText(img, labels[np.argmax(p)], (x1, y1),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        name = f.split('/')[-1]
        cv2.imwrite('./generate/predict/'+name, img)
