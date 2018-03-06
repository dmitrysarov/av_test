import click
from my_misc import *
import tensorflow as tf
from tensorflow.python.saved_model  import tag_constants
from skimage import io

EXPORT_DIR = 'model'
@click.command()
@click.option('--path', default = 'test_image.jpg', help = 'path to image')
def main(path):
    images = io.imread(path) #TODO images batch handling
    image_shape = images.shape
    molded_images, image_metas, windows = mold_inputs([images])
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], EXPORT_DIR)
        detections = sess.run('mrcnn_detection/Reshape_1:0',
                                {'input_image:0': molded_images,
                                'input_image_meta:0': image_metas})
    for i, image in enumerate([images]):
        boxes, class_ids, scores = unmold_detections(detections[i],
                                                           image_shape,
                                                           windows[i])
        if boxes.shape[0] == 0 :
            print('there is no detected object')
            pred_box = [0,0] + list(image_shape[:2])
        else:
            # group detected objects
            pred_box = np.min(boxes[:,:2], axis = 0).tolist()[::-1] + \
                np.max(boxes[:,2:], axis = 0).tolist()[::-1]
            pred_box /= np.array([image_shape[1],image_shape[0],
                                 image_shape[1],image_shape[0]])
        print(pred_box)
if __name__ == '__main__':
    main()
