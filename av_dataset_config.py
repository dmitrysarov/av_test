import sys
from skimage import io
import copy
import numpy as np
from PIL import Image, ImageDraw
sys.path.append('../Mask_RCNN/')
from utils import Dataset
from config import Config
import logging
logging.basicConfig(level=logging.INFO,
format='%(levelname)s: %(asctime)s ::: %(name)s: %(message)s (%(filename)s:%(lineno)d)',
                    datefmt='%Y-%m-%d %H:%M:%S')


class AvConfig(Config):
    """
    """
    # Give the configuration a recognizable name
    NAME = "av"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 192
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 8

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 800

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200


def vert_to_mask(points, width, height):
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).rectangle(points, outline=1, fill=1)
    return np.array(img)


class AvDataset(Dataset):
    def load_dataset(self, dataset):
        '''
        dataset  [{'path':'path/to/image',
                   'vertices':[x1,y1,x2,y2]}
        '''
        self.add_class("av", 1, "object")
        for num, example in enumerate(dataset):
            example_ = copy.deepcopy(example)
            image = io.imread(example['path'])
            height, width, _ = image.shape
            example_.update({'height': height,
                             'width': width})
            self.add_image("av", image_id=num, **example_)

    def load_mask(self, image_id):
        '''
        Generate instance masks for shapes of the given image ID.
        '''
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width']], dtype=np.uint8)
        # move from relative to absolute coord
        mask = vert_to_mask(info['vertices'], info['width'], info['height'])
        return np.expand_dims(mask, 2), np.array([1], dtype=np.int32)
