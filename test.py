# Local files
import utils
import models

# Utils cloned from tensorflow/models
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import colab_utils


import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# configure plot settings via rcParams
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labelsize'] = False
plt.rcParams['ytick.labelsize'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.right'] = False



# Build model
detection_model = models.build_detection_model()


@tf.function(jit_compile=True)
def detect(input_tensor):
    """Run detection on an input image.

    Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

    Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
    """

    # Preprocess
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)

    # Predict
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    
    # use the detection model's postprocess() method to get the the final detections
    detections = detection_model.postprocess(prediction_dict, shapes)
    
    return detections



# Define a dictionary describing the zombie class (used for viz util)
category_index = {1: {'id': 1, 'name': 'zombie'}}

# Offset from zero for our class names (1 in this case, since we have a class of label 1)
label_id_offset = 1



if __name__ == "__main__":

    # Get list of image paths
    image_paths = glob.glob(test_image_dir + "zombie-walk*.jpg")
    
    # Load images into numpy array
    test_images_np = utils.load_image_into_numpy_array(image_paths)

    # Dict to hold results
    results = {'boxes': [], 'scores': []}

    for i, image in enumerate(test_images_np):

        # Convert to tensor
        input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

        # Preprocess, predict on, and postprocess the image
        detections = detect(input_tensor)[0]

        # Visualize results
        plot_detections(
            image_np = image[0],
            boxes = detections['detection_boxes'].numpy(),
            classes = detections['detection_classes'].numpy().astype(np.uint32) - label_id_offset,
            scores = detections['detection_scores'].numpy(),
            image_name=f"./results/gif_frame_{i:03}.jpg",
            category_index = category_index,
            figsize=(15, 20), 
        )