from config import Config
from scipy.misc import  imresize
import numpy as np


class InferenceConfig(Config):
    NAME = "ma"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1

config = InferenceConfig()

def mold_image(images, config):
    return images.astype(np.float32) - config.MEAN_PIXEL

def mold_inputs(images):
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        molded_image, window, scale, padding = resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        molded_image = mold_image(molded_image, config)
        image_meta = np.array(
            [0] + list(image.shape) +
            list(window) +
            list(np.zeros([config.NUM_CLASSES], dtype=np.int32)))
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

def unmold_detections(detections, image_shape, window):
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    h_scale = float(image_shape[0]) / (window[2] - window[0])
    w_scale = float(image_shape[1]) / (window[3] - window[1])
    scale = min(h_scale, w_scale)
    shift = window[:2]  # y, x
    scales = np.array([scale, scale, scale, scale])
    shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
    boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
    return boxes, class_ids, scores

def resize_image(image, min_dim=None, max_dim=None, padding=False):
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    if min_dim:
        scale = max(1, (min_dim) / min(h, w))
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = (max_dim) / image_max
    if scale != 1:
        image = imresize(image, (round(h * scale),
                                 round(w * scale)))
       # image = resize(
       #     image, (round(h * scale), round(w * scale)), mode = 'constant')
    if padding:
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding
