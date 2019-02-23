+++
title = "Jupyter Notebook"
subtitle = "tf.keras for mnist"

# Add a summary to display on homepage (optional).
summary = "Tensorflow tutorial"

date = 2019-02-23T15:14:50+08:00
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["Sangjee Dondrub"]

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = []
categories = []

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["deep-learning"]` references
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
# projects = ["internal-project"]

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
[image]
  # Caption (optional)
  caption = ""

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
+++


```python
import tensorflow as tf
from tensorflow import keras
```


```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
```

    /Users/sangjee/anaconda/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


    1.12.0



```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```


```python
class_names = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']
```


```python
train_images.shape
```




    (60000, 28, 28)




```python
len(train_images)
```




    60000




```python
train_labels
```




    array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)




```python
test_images.shape
```




    (10000, 28, 28)




```python
len(test_images)
```




    10000



## Process the data

The original data's pixel values range from `0` to `255`, indicating the data is with color. Let's see the first image in the train set


```python
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
```


![png](./tutorial_9_0.png)


Therefore we scale those values to range of `0` to `1` before feeding into the nerual network model. For this cast the datatype of the image compoenents from an integer to float, and divide bu 255.


```python
train_images = train_images / 255
test_images = test_images / 255
```

Let's see the first `25` images from training set and display the class for each image


```python
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
```


![png](./tutorial_13_0.png)


# Build the model

## Setup layers


```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
```

## Compile the model


```python
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Fitting the model with training data


```python
tensorboard = keras.callbacks.TensorBoard(log_dir='/tmp/mnist-simple-dense')
```


```python
model.fit(
    train_images,
    train_labels,
    epochs=5,
    callbacks=[tensorboard]
)
```

    Epoch 1/5
    60000/60000 [==============================] - 4s 73us/step - loss: 0.4974 - acc: 0.8264
    Epoch 2/5
    60000/60000 [==============================] - 4s 69us/step - loss: 0.3728 - acc: 0.8676
    Epoch 3/5
    60000/60000 [==============================] - 4s 70us/step - loss: 0.3372 - acc: 0.8780
    Epoch 4/5
    60000/60000 [==============================] - 4s 72us/step - loss: 0.3118 - acc: 0.8864
    Epoch 5/5
    60000/60000 [==============================] - 5s 80us/step - loss: 0.2961 - acc: 0.8916

    <tensorflow.python.keras.callbacks.History at 0x1277497b8>


## Evaluate accuracy


```python
test_loss, test_acc = model.evaluate(
    test_images,
    test_labels
)
```

    10000/10000 [==============================] - 0s 44us/step



```python
print('Test Accuracy: ', test_acc)
```

    Test Accuracy:  0.8745


## Make prediction


```python
predictions = model.predict(test_images)
```


```python
np.argmax(predictions[0])
```




    9




```python
test_labels[0]
```




    9



## Let's do some graphing


```python
def plot_image(i,predictions_array,true_label,img):
    predictions_array, true_label,img = predictions_array[i],true_label[i],img[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img,cap=plt.cm.binary)
```


```python
# end https://www.tensorflow.org/tutorials/keras/basic_classification
```
