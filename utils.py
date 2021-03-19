# Utilities cloned from tensorflow/models
from object_detection.utils import visualization_utils

from PIL import Image, ImageDraw, ImageFont
from six import BytesIO
import glob

import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np



def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    
    # File I/O without threadlocking
    img_data = tf.io.gfile.GFile(path, 'rb').read()    

    # Open image
    image = Image.open(BytesIO(img_data))
    
    # Get image size
    (im_width, im_height) = image.size
    
    # Load into array, reshape, cast to 8-bit integer 
    img_array = np.array(image.getdata()) \
        .reshape((im_height, im_width, 3)) \
        .astype(np.uint8)
    
    return img_array



def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
    """Wrapper function to visualize detections.

    Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
          and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
          this function assumes that the boxes to be plotted are groundtruth
          boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
          category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
    """
    
    image_np_with_annotations = image_np.copy()
    
    visualization_utils \
        .visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    
    else:
        plt.imshow(image_np_with_annotations)


def load_images_into_numpy_array(image_paths)
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    # declare an empty list
    images = []
    
    for image_path in image_paths

        image_array = load_image_into_numpy_array(image_path)

        images.append(image_array)
        
    return images


def prepare_images_boxes_and_classes_as_tensors(images_np, 
                                                boxes_np, 
                                                classes_np, 
                                                label_id_offset=1):
    """Prepares image, classes, and boxes for object detection
    by converting from numpy arrays into tf.Tensor objects

    args:
        images_np : image array, 
        boxes_np : array of ground truth bounding boxes, 
        classes: array of image classification labels, 
        label_id_offset=1
    """

    if classes_np == None:        
        classes_np = np.ones(shape=[box.shape[0]], dtype=np.int32)


    image_tensors = []
    box_tensors = []
    classes_tensors = []

    for image, box, classes in zip(images_np, boxes_np, classes_np):

        # convert training image to tensor, add batch dimension, and add to list
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensors.append(tf.expand_dims(image_tensor, axis=0))

        # convert numpy array to tensor, then add to list
        box_tensor = tf.convert_to_tensor(box, dtype=tf.float32)
        box_tensors.append(box_tensor)

        # apply offset to to have zero-indexed ground truth classes
        classes_minus_offset = tf.convert_to_tensor(classes - label_id_offset)
        # do one-hot encoding to ground truth classes
        classes_tensor = tf.one_hot(classes_minus_offset, num_classes)
        classes_tensors.append(classes_tensor)

    return image_tensors, box_tensors, classes_tensors



