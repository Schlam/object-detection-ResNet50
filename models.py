import tensorflow as tf
tf.keras.backend.clear_session()

# Packages from within this repository
from object_detection.utils import config_util
from object_detection.builders import model_builder


# Path to downloaded model checkpoint
checkpoint_path = 'models/research/object_detection/test_data/checkpoint/ckpt-0'


# Define the path to the .config file for ssd resnet 50 v1 640x640
pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'



def build_detection_model(checkpoint_path='models/research/object_detection/test_data/checkpoint/ckpt-0',
                          pipeline_config='models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config',
                          freeze_batchnorm=True,
                          num_classes=1):
    """Build an object detection model from ResNet50 architecture pretrained on COCO
    
    Args:
        
        pipeline_config - path to .config file

        checkpoint_path - path to checkpoint.index file, not including .index

    """

    # Load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)

    # Read in the object stored at the key 'model' of the configs dictionary
    model_config = configs['model']

    # Modify the number of classes from its default of 90
    model_config.ssd.num_classes = num_classes

    # Freeze batch normalization
    model_config.ssd.freeze_batchnorm = freeze_batchnorm

    # Build model
    detection_model = model_builder.build(model_config=model_config, is_training=True)



    # Bounding box prediction layers
    tmp_box_predictor_checkpoint = tf.train.Checkpoint(    
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        _box_prediction_head=detection_model._box_predictor._box_prediction_head)

    # Model checkpoint
    model_checkpoint = tf.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=tmp_box_predictor_checkpoint
    )


    # Define a checkpoint that sets model equal to the model defined above
    checkpoint = tf.train.Checkpoint(model=model_checkpoint)

    # Restore the checkpoint to the checkpoint path
    checkpoint.restore(checkpoint_path)

    # use the detection model's `preprocess()` method and pass a dummy image
    tmp_image, tmp_shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))

    # run a prediction with the preprocessed image and shapes
    tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)

    # postprocess the predictions into final detections
    _ = detection_model.postprocess(tmp_prediction_dict, tmp_shapes)

    return detection_model