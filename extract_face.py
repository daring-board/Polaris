import os
from mtcnn.mtcnn import MTCNN
base_path = '/content/drive/My Drive/share/kanjani/YY/'
dist_path = '/content/drive/My Drive/share/kanjani/resultYY/'
f_list = os.listdir(base_path)

detector = MTCNN()
for f in f_list:
  print(f)
  if not os.path.isfile(base_path + f):
    continue
  comp = f.split('.')
  im = cv2.imread(base_path + f)
  img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  results = detector.detect_faces(img)

  for idx, item in enumerate(results):
    x1, y1, width, height = item['box']
    cv2.rectangle(img, (x1,y1), (x1+width, y1+height), (255, 0, 0), 2)
    cv2.imwrite(dist_path + comp[0] + '_%d.jpeg'%idx, im[y1: y1+height, x1: x1+width,:])