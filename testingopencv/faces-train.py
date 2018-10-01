import os
import pickle
from PIL import Image
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'faceimgs')
# print image_dir
face_cascade_frontalface_alt2 = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels, x_train = [], []

for root, dirs, files in os.walk(image_dir):
    for f in files:
        # print "seaching over ", f
        if f.endswith(".PNG"):
            path = os.path.join(root, f)
            label = os.path.basename(f)#.replace(" ", "-")
            label = label.split(' ')[0]
            #label = os.path.basename(path)#(os.path.dirname(path))#  .replace(" ", "-")
            print path," is labeled as ", label
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print label_ids

            pil_image = Image.open(path).convert("L")  # grayscale
            size = (550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, 'uint8')
            # print image_array
            faces = face_cascade_frontalface_alt2.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=1)

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

