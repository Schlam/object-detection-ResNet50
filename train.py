import glob


# Local files
import utils
import models

import tensorflow as tf
import numpy as np





"""Contained within the code in this repository is everything needed to

- Build an object detection model with some pre-existing architechture
- Initialize that model with pretrained weights from a given checkpoint
- Fine tune specific layers of said model for relevant downstream tasks



The current use case is to build a "Zombie detector," which makes for a
simple fine-tuning example given that our model only needs 
to be trained on one additional class, and the layers 
involved with region proposal can be left alone.


This is reflected by the values for `prefixes` and `variables,`
as the variables selected to be updated through training are 
only those that have names starting with one of the two strings:


1. 'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead'
2. 'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead'




To customize this script, consider updating the following arguments to train()

 `model` - Pre-compiled object detection model
 `dataset` - List of tensors for images, boxes and classes
 `batch_size` - Batch size and number of training epochs
 `variables` - List of variable names to be updated during training
 `optimizer` - Optimizer 


"""




@tf.function(jit_compile=True)
def training_step(data
                  model,
                  optimizer,
                  variables):
    """A single training iteration.

    Args:

        training_data: A tuple of the three following lists:
            Image tensors,  a list of [1, height, width, 3] Tensor of type tf.float32.
                Note that the height and width can vary across images, as they are
                reshaped within this function to be 640x640.
            Ground truth bounding boxes, a list of Tensors of shape [N_i, 4] with type
                tf.float32 representing groundtruth boxes for each image in the batch.
            Ground truth class labels, a list of Tensors of shape [N_i, num_classes]
                with type tf.float32 representing groundtruth boxes for each image in
                the batch.

        model: model to be trained
    
        optimizer: optimizer used during training

        variables: model variables which are updated through training 


    Returns:

      A scalar tensor representing the total loss for the input batch.
    
    """

    # Get images, boxes, and classes from data tuple
    (image_list, groundtruth_boxes_list, groundtruth_classes_list) = data
        
    # Provide the ground truth to the model
    detection_model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_boxes_list,
        groundtruth_classes_list=groundtruth_classes_list)


    preprocessed_image_list = []
    image_shapes_list = []
    
    with tf.GradientTape() as tape:

        for image in image_list:

            # Preprocess image
            preprocessed_image, original_image_shape = detection_model.preprocess(image)
            
            # Append image and shape tensors to lists
            preprocessed_image_list.append(preprocessed_image)
            image_shapes_list.append(original_image_shape)

        # Get tensors from images, shapes
        images = tf.concat(preprocessed_image_list, axis=0)
        shapes = tf.concat(image_shapes_list, axis=0)
        
        # Predict on the data
        prediction_dict = detection_model.predict(images, shapes)

        # Calculate the total loss (sum of both losses)
        losses_dict = model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

    # Calculate gradients, update model parameters
    gradients = tape.gradient(total_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

        
    return total_loss



@tf.function(jit_compile=True)
def train(model, 
          dataset,
          variables,
          batch_size,
          optimizer,
          epochs=1):
    """Performs model training for a given dataset

    Args:
        model: model to be trained, only supports ResNet at the moment

        dataset: A tuple of the three following lists:
            image_tensors: a list of [1, height, width, 3] Tensor of type tf.float32.
                Note that the height and width can vary across images, as they are
                reshaped within this function to be 640x640.
            groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
                tf.float32 representing groundtruth boxes for each image in the batch.
            groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
                with type tf.float32 representing groundtruth boxes for each image in
                the batch.

        variables: model variable names which are to be updated during training 

        batch_size: size of training batch, should be denominator of dataset length

        optimizer: optimizer used during training

        epochs: number of training epochs

    Returns:

        A scalar tensor representing the total loss for the input batch.

    """

    # History to hold loss
    history = {"loss":[]}

    # Create a list of indices for shuffling
    shuffled_keys = list(range(len(dataset)))

    # Get number of batches by taking the number of samples over batch size (add 1 if remainder)
    batches = (len(dataset) // batch_size + 1 * bool(len(dataset) % batch_size))
    
    for _ in range(epochs):


        # Shuffle index 
        np.random.shuffle(shuffled_keys)


        for batch in range(batches):

            # Slice shuffled indices according to batch size and current batch
            keys = shuffled_keys[batch * batch_size, (1 + batch) * batch_size]

            # Training step (forward pass + backwards pass)
            total_loss = training_step(dataset[keys], model, optimizer, variables)

            # Add loss to history 
            history['loss'] += [total_loss.numpy()]

            # Print out an update on current training progression
            print(f"Batch {batch} of {batches}\nloss = {total_loss.numpy()}")    
    
    
    return history




# Base directory, helps shorten path variables
base_dir = "models/research/object_detection/"

# Build model with desired configuration
detection_model = models.build_detection_model(
    checkpoint_path = base_dir + 'test_data/checkpoint/ckpt-0',
    pipeline_config = base_dir + 'configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
)



# Define Paths to each image file, and load into numpy arrays    
image_paths = glob.glob("training/training-zombie*.jpg")
training_image_arrays = utils.load_images_into_numpy_array(image_paths)

# Define bounding boxes for our training images
groundtruth_boxes = np.array([
    [[0.27333333, 0.41500586, 0.74333333, 0.57678781]],
    [[0.29833333, 0.45955451, 0.75666667, 0.61078546]],
    [[0.40833333, 0.18288394, 0.945, 0.34818288]],
    [[0.16166667, 0.61899179, 0.8, 0.91910903]],
    [[0.28833333, 0.12543962, 0.835, 0.35052755]]]    
)

# Define class labels for our training images
groundtruth_classes = np.ones(shape=[box.shape[0]], dtype=np.int32)

# Get tensors for model training
training_data = utils.images_boxes_and_classes_as_tensors(images_np=training_image_arrays, 
                                                          boxes_np=groundtruth_boxes,
                                                          classes_np=groundtruth_classes)



# Prefixes for variable names we wish to tune
prefixes = ['WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']

# Names for variables to be updated during training
variables = [var for var in detection_model.trainable_variables
             if any([var.name.startswith(prefix) for prefix in prefixes])]



# Hyperparameters
batch_size = 4
epochs = 1

# Optimizer
optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)




if __name__ == "__main__":

    train(

        model = detection_model,
        dataset = training_data,
        batch_size = batch_size,
        variables = variables,
        optimizer = optimizer,
        epochs = epochs

    )