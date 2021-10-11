from core import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default="output/test_paths.txt", help="Path to input")
args = vars(ap.parse_args())

image_paths = open(args["input"]).read().strip().split("\n")

print("[INFO] Loading Object Detectors...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.LB_PATH, "rb").read())

for image_path in image_paths:
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    box_preds, label_preds = model.predict(image)
    start_x, start_y, end_x, end_y = box_preds[0]
    
    i = np.argmax(label_preds, axis=1)
    label = lb.classes_[i][0]
    
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    h, w = image.shape[:2]
    
    start_x = int(start_x * w)
    start_y = int(start_y * h)
    end_x = int(end_x * w)
    end_y = int(end_y * h)
    
    y = start_y - 10 if start_y - 10 > 10 else start_y + 10
    cv2.putText(image, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    cv2.imshow("Output", image)
    cv2.waitKey()
cv2.destroyAllWindows()
