# Clone TensorFlow/Models repo
git clone --depth 1 https://github.com/tensorflow/models/
# 
# Cd into models/research
cd models/research/
# 
# Parse protocol buffer (not entirely sure why if I'm being honest)
protoc object_detection/protos/*.proto --python_out=.
# 
# Copy setup TF2 setup script from models/research/object_detection
cp object_detection/packages/tf2/setup.py .
# 
# Install that jawn
python -m pip install .
# 
# 
# 
# I would highly suggest you do not run this next command if I'm being real with you, 
# get your own images and use the tools provided in the Object Detection API for labelling.
# Literally 3 lines of code and you can have enough data for few-shot learning
# 
# example:
# 
# ```python
# 
# from object_detection.utils import colab_utils
# 
# 
# Empty list to hold bounding boxes
# ground_truth_boxes = []
# 
# # Use GUI annotation tool in CoLab to quickly get target values
# colab_utils.annotate(train_images_np, box_storage_pointer=gt_boxes)
# 
# ```
#
#
wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training-zombie.zip -O ./training-zombie.zip
# 
# (w)get the model checkpoints
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
# Uzippity zip that ish crip 
tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
# 
# It's time for you to move on
mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint models/research/object_detection/test_data/