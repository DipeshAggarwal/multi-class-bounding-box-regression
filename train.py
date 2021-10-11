from core import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

print("[INFO] Loading Dataset...")
data = []
labels = []
bounding_boxes = []
image_paths = []

for csv_path in paths.list_files(config.ANNOTS_PATH, validExts=(".csv")):
    rows = open(csv_path).read().strip().split("\n")
    
    for row in rows:
        row = row.split(",")
        filename, start_x, start_y, end_x, end_y, label = row
        
        image_path = os.path.sep.join([config.IMAGES_PATH, label, filename])
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        start_x = float(start_x) / w
        start_y = float(start_y) / h
        end_x = float(end_x) / w
        end_y = float(end_y) / h
        
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        
        data.append(image)
        labels.append(label)
        bounding_boxes.append((start_x, start_y, end_x, end_y))
        image_paths.append(image_path)
        
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bounding_boxes = np.array(bounding_boxes, dtype="float32")
image_paths = np.array(image_paths)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

if len(lb.classes_) == 2:
    labels = to_categorical(labels)

split = train_test_split(data, labels, bounding_boxes, image_paths, test_size=0.2, random_state=42)

train_images, test_images = split[:2]
train_labels, test_labels = split[2:4]
train_bboxes, test_bboxes = split[4:6]
train_paths, test_paths = split[6:]

print("[INFO] Saving Testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(test_paths))
f.close()

vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
vgg.trainable = False

x = vgg.output
x = Flatten()(x)

bb_head = Dense(128, activation="relu")(x)
bb_head = Dense(64, activation="relu")(bb_head)
bb_head = Dense(32, activation="relu")(bb_head)
bb_head = Dense(4, activation="sigmoid", name="bounding_box")(bb_head)

class_head = Dense(512, activation="relu")(x)
class_head = Dropout(0.5)(class_head)
class_head = Dense(512, activation="relu")(class_head)
class_head = Dropout(0.5)(class_head)
class_head = Dense(len(lb.classes_), activation="softmax", name="class_label")(class_head)

model = Model(inputs=vgg.input, outputs=(bb_head, class_head))

losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error"
    }

loss_weights = {
    "class_label": 1.0,
    "bounding_box": 1.0
    }

opt = Adam(lr=config.INIT_LR)
model.compile(loss=losses, metrics=["accuracy"], loss_weights=loss_weights)
print(model.summary())

train_targets = {
    "class_label": train_labels,
    "bounding_box": train_bboxes
    }
test_targets = {
    "class_label": test_labels,
    "bounding_box": test_bboxes
    }

print("[INFO] Training Model...")
H = model.fit(
    train_images, train_targets,
    validation_data=(test_images, test_targets),
    batch_size=config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    verbose=1)

print("[INFO] Saving Object Detection Model...")
model.save(config.MODEL_PATH, save_format="h5")

print("[INFO] Saving Label Binarizer")
f = open(config.LB_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

loss_names = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
fig, ax = plt.subplots(3, 1, figsize=(13, 13))

for i, l in enumerate(loss_names):
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], label="val_" + l)
    ax[i].legend()
    
plt.tight_layout()
plot_path = os.path.sep.join([config.PLOTS_PATH, "losses.png"])
plt.savefig(plot_path)
plt.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"], label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"], label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

plotPath = os.path.sep.join([config.PLOTS_PATH, "accs.png"])
plt.savefig(plot_path)
