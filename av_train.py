import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

import sys
sys.path.append('../Mask_RCNN/')

import utils
import model as modellib
from av_dataset_config import AvConfig, AvDataset
import pandas as pd

ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR+'/av_test', "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

# get model
config = AvConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
# TODO try imagenet weights
# model.load_weights(model.get_imagenet_weights(), by_name=True)
model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])
# get dataset
df = pd.read_csv('av_test/with_labels_abs.csv')
df_train = df.sample(frac=0.8, random_state=200)
df_val = df.drop(df_train.index)
# train
dataset = []
for index, row in df_train.iterrows():
    dataset.append({'path': 'av_test/images/'+row['image_name'],
                    'vertices': [row['x1'], row['y1'],
                                 row['x2'], row['y2']]})
data_train = AvDataset()
data_train.load_dataset(dataset)
data_train.prepare()
# val
dataset = []
for index, row in df_val.iterrows():
    dataset.append({'path': 'av_test/images/'+row['image_name'],
                    'vertices': [row['x1'], row['y1'],
                                 row['x2'], row['y2']]})
data_val = AvDataset()
data_val.load_dataset(dataset)
data_val.prepare()

# start training
model.train(data_train, data_val, learning_rate=config.LEARNING_RATE, epochs=1,
            layers='heads')
