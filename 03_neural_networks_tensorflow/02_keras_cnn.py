# %% [markdown]
# # CIFAR-10 dataset classification with CNNs
# 
# Author: Tanwi Mallick, adapting codes from Bethany Lusch, Prasanna Balprakash, Corey Adams, and Kyle Felker
# 
# In this notebook, we'll continue the CIFAR-10 problem using the Keras API (as included in the TensorFlow library) and incorporating convolutional layers.
# 
# First, the needed imports.

# %%
# %matplotlib inline

import tensorflow as tf

import numpy
import matplotlib.pyplot as plt
import time

# Fixes issue with Tensorflow crashing?
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %% [markdown]
# ## CIFAR-10 data set
# 
# Again we'll load the cifar10 data set. CIFAR-10 dataset contains 32x32 color images from 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. If you haven't downloaded it already, it could take a while.

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype(numpy.float32)
x_test  = x_test.astype(numpy.float32)

x_train /= 255.
x_test  /= 255.

y_train = y_train.astype(numpy.int32)
y_test  = y_test.astype(numpy.int32)

# %% [markdown]
# This time we won't flatten the images. 
# 
# The training data (`X_train`) is a 3rd-order tensor of size (50000, 32, 32), i.e. it consists of 50000 images of size 32x32 pixels. 
# 
# `y_train` is a 50000-dimensional vector containing the correct classes ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') for each training sample.

# %% [markdown]
# ## Convolutional neural network (CNN)
# 
# CNN is a type of deep learning model for processing data that has a grid pattern, such as images.
# 
# Let's use a small model that includes convolutional layers
# 
# - The Conv2D layers operate on 2D matrices so we input the digit images directly to the model.
#     - The two Conv2D layers belows learn 32 and 64 filters respectively. 
#     - They are learning filters for 3x3 windows.
# - The MaxPooling2D layer reduces the spatial dimensions, that is, makes the image smaller.
#     - It downsamples by taking the maximum value in the window 
#     - The pool size of (2, 2) below means the windows are 2x2. 
#     - Helps in extracting important features and reduce computation
# - The Flatten layer flattens the 2D matrices into vectors, so we can then switch to Dense layers as in the MLP model.
# 
# See https://keras.io/layers/convolutional/, https://keras.io/layers/pooling/ for more information.

# %% [markdown]
# ![conv layer](images/conv_layer.png)
# Image credit: [Jason Brownlee](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)

# %% [markdown]
# ![conv layer](images/conv.png)
# Image credit: [Anh H. Reynolds](https://anhreynolds.com/blogs/cnn.html)

# %% [markdown]
# 
# <img src="images/MaxpoolSample2.png" width="600" hight="600" align="left"/>

# %%
class CIFAR10Classifier(tf.keras.models.Model):

    def __init__(self, activation=tf.nn.tanh):
        tf.keras.models.Model.__init__(self)

        # For example: the first layer learns 32 3x3 kernels that could e.g. detect edges or color transitions.
        self.conv_1 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu')
        self.pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.drop_4 = tf.keras.layers.Dropout(0.25)
        self.dense_5 = tf.keras.layers.Dense(128, activation='relu')
        self.drop_6 = tf.keras.layers.Dropout(0.5)
        self.dense_7 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):

        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.pool_3(x)
        x = self.drop_4(x)
        # Flatten because we will be feeding to dense layers next.
        x = tf.keras.layers.Flatten()(x)
        x = self.dense_5(x)
        x = self.drop_6(x)
        x = self.dense_7(x)

        return x

# %% [markdown]
# ### Simple training

# %% [markdown]
# Here is a concise way to train the network, like we did in the previous notebook. We'll see a more verbose approach below that allows more performance tuning.

# %%
def train_network_concise(_batch_size, _n_training_epochs, _lr):

    cnn_model = CIFAR10Classifier()

    cnn_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    history = cnn_model.fit(x_train, y_train, batch_size=_batch_size, epochs=_n_training_epochs)
    return history, cnn_model

# %%
# This took 55 seconds per epoch on my laptop
batch_size = 512
epochs = 3
lr = .01
history, cnn_model = train_network_concise(batch_size, epochs, lr)

# %% [markdown]
# Accuracy for test data.  The model should be better than the non-convolutional model even if you're only patient enough for three epochs. 

# %%
plt.figure(figsize=(5,3))
plt.plot(history.epoch, history.history['loss'])
plt.title('loss')

plt.figure(figsize=(5,3))
plt.plot(history.epoch, history.history['accuracy'])
plt.title('accuracy')

# %% [markdown]
# ### Inference

# %% [markdown]
# With enough training epochs, the test accuracy should exceed 99%.
# 
# You can compare your result with the state-of-the art [here](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html). Even more results can be found [here](http://yann.lecun.com/exdb/mnist/).

# %%
x_test_reshaped = numpy.expand_dims(x_test, -1)
scores = cnn_model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (cnn_model.metrics_names[1], scores[1]*100))

# %% [markdown]
# We can also again check the confusion matrix

# %%
from sklearn.metrics import confusion_matrix

print('Confusion matrix (rows: true classes; columns: predicted classes):')
print()
predictions = cnn_model.predict(x_test)
cm=confusion_matrix(y_test, numpy.argmax(predictions, axis=1), labels=list(range(10)))
print(cm)
print()

print('Classification accuracy for each class:')
print()
for i,j in enumerate(cm.diagonal()/cm.sum(axis=1)): print("%d: %.4f" % (i,j))

# %% [markdown]
# ### More verbose training

# %% [markdown]
# This approach explicitly handles the looping over data. It will be helpful this afternoon for diving in and optimizing

# %%
def compute_loss(y_true, y_pred):
    # if labels are integers, use sparse categorical crossentropy
    # network's final layer is softmax, so from_logtis=False
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # if labels are one-hot encoded, use standard crossentropy

    return scce(y_true, y_pred)  

# %%
def forward_pass(model, batch_data, y_true):
    y_pred = model(batch_data)
    loss = compute_loss(y_true, y_pred)
    return loss

# %%
# Here is a function that will manage the training loop for us:

def train_loop(dataset, batch_size, n_training_epochs, model, opt):
    
    @tf.function()
    def train_iteration(data, y_true, model, opt):
        with tf.GradientTape() as tape:
            loss = forward_pass(model, data, y_true)

        trainable_vars = model.trainable_variables

        # Apply the update to the network (one at a time):
        grads = tape.gradient(loss, trainable_vars)

        opt.apply_gradients(zip(grads, trainable_vars))
        return loss

    for i_epoch in range(n_training_epochs):
        print("beginning epoch %d" % i_epoch)
        start = time.time()

        epoch_steps = int(50000/batch_size)
        dataset.shuffle(50000) # Shuffle the whole dataset in memory
        batches = dataset.batch(batch_size=batch_size, drop_remainder=True)
        
        for i_batch, (batch_data, y_true) in enumerate(batches):
            batch_data = tf.reshape(batch_data, [-1, 32, 32, 3])
            loss = train_iteration(batch_data, y_true, model, opt)
            
        end = time.time()
        print("took %1.1f seconds for epoch #%d" % (end-start, i_epoch))

# %%
def train_network(dataset, _batch_size, _n_training_epochs, _lr):

    mnist_model = CIFAR10Classifier()

    opt = tf.keras.optimizers.Adam(_lr)

    train_loop(dataset, _batch_size, _n_training_epochs, mnist_model, opt)
    return mnist_model

# %%
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset.shuffle(50000)

batch_size = 512
epochs = 3
lr = .01
model = train_network(dataset, batch_size, epochs, lr)

# %%
print('Confusion matrix (rows: true classes; columns: predicted classes):')
print()
predictions = model.predict(x_test)
cm=confusion_matrix(y_test, numpy.argmax(predictions, axis=1), labels=list(range(10)))
print(cm)
print()

print('Classification accuracy for each class:')
print()
for i,j in enumerate(cm.diagonal()/cm.sum(axis=1)): print("%d: %.4f" % (i,j))

# %% [markdown]
# # Homework: improve the accuracy of this model

# %% [markdown]
# Update this notebook to ensure more accuracy. How high can it be raised? Changes like increasing the number of epochs, altering the learning weight, altering the number of neurons the hidden layer, chnaging the optimizer, etc. could be made directly in the notebook. You can also change the model specification by expanding the network's layer. The current notebook's training accuracy is roughly 58.69%, although it varies randomly.

# %%



