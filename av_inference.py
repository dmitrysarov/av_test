import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))


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
MODEL_DIR = os.path.join(ROOT_DIR + '/av_test', "logs")



class InferenceConfig(AvConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
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
for index, row in df_val.iterrows():
    path = 'av_test/images/'+row['image_name']
    image = io.imread(path)
    results = model.detect([image], verbose=0)[0]['rois']
    # combine found obkects
    pred_x1, pred_y1, _, _ = np.min(results, axis = 0)
    _, _, pred_x2, pred_y2 = np.max(results, axis = 0)
    pred_roi = [pred_x1, pred_y1, pred_x2, pred_y2]
    gt_roi = [row['x1'], row['y1'], row['x2'], row['y2']]
    iou.append(bb_intersection_over_union(pred_roi, gt_roi))

logging.info('average iou is {}'.format(np.mean(iou)))
