import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))


import cv2
import numpy as np
import pandas as pd
from skimage import io
import logging
logging.basicConfig(level=logging.INFO,
format='%(levelname)s: %(asctime)s ::: %(name)s: %(message)s (%(filename)s:%(lineno)d)',
                    datefmt='%Y-%m-%d %H:%M:%S')
import sys
sys.path.append('../Mask_RCNN/')
import model as modellib
from av_dataset_config import AvConfig

ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class InferenceConfig(AvConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# get model
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model_path = model.find_last()[1]
model.load_weights(model_path, by_name=True)
# get dataset
df = pd.read_csv('av_test/with_labels_abs.csv')
df_train = df.sample(frac=0.8, random_state=200)
df_val = df.drop(df_train.index)
# predict validation
iou = []
if not os.path.isdir('prediction'):
    os.makedirs('prediction')
for index, row in df_val.iterrows():
    path = 'av_test/images/'+row['image_name']
    image = io.imread(path)
    results = model.detect([image], verbose=0)[0]['rois']
    # combine found objects
    pred_x1, pred_y1, _, _ = np.min(results, axis = 0).astype(int).astype(int)
    _, _, pred_x2, pred_y2 = np.max(results, axis = 0).astype(int).astype(int)
    pred_roi = [pred_x1, pred_y1, pred_x2, pred_y2]
    gt_roi = [row['x1'], row['y1'], row['x2'], row['y2']]
    gt_roi = [int(x) for x in gt_roi]
    cv2.rectangle(image, tuple(pred_roi[:2]), tuple(pred_roi[2:]), (0, 255, 0), 5)
    cv2.rectangle(image, tuple(gt_roi[:2]), tuple(gt_roi[2:]), (255, 0, 0), 5)
    cv2.imwrite('prediction/' + row['image_name'], image)
