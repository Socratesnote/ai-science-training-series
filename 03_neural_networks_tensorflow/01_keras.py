# %% [markdown]
# # CIFAR-10 dataset classification with MLPs
# 
# Author: Tanwi Mallick, adapting codes from Bethany Lusch, Prasanna Balprakash, Corey Adams, and Kyle Felker
# 
# In this notebook, we'll continue the CIFAR-10 problem but using the Keras API (as included in the TensorFlow library)
# 
# First, the needed imports.

# %%
%matplotlib inline

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
# 
# <img src="images/CIFAR-10.png"  align="left"/>

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype(numpy.float32)
x_test  = x_test.astype(numpy.float32)

x_train /= 255.
x_test  /= 255.

y_train = y_train.astype(numpy.int32)
y_test  = y_test.astype(numpy.int32)

print()
print('CIFAR-10 data loaded: train:',len(x_train),'test:',len(x_test))
print('X_train:', x_train.shape)
print('y_train:', y_train.shape)

# %% [markdown]
# ### Download the dataset and load

# %%
# Alternative download and unzipping.
# !pip install image-dataset-loader
# !wget https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
# !tar -xf cifar10.tgz

# %%
if False:
    %pip install image-dataset-loader
    %pip install wget
    import wget
    wget.download('https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz')
    # !tar -xf cifar10.tgz

    # Or
    !https_proxy=http://proxy.tmi.alcf.anl.gov:3128  pip install image-dataset-loader
    !https_proxy=http://proxy.tmi.alcf.anl.gov:3128  wget https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
    !tar -xf cifar10.tgz


# %%
# Show alternative data.
if False:
    from image_dataset_loader import load
    (x_train, y_train), (x_test, y_test) = load('cifar10', ['train', 'test'])

    x_train = x_train.astype(numpy.float32)
    x_test  = x_test.astype(numpy.float32)

    x_train /= 255.
    x_test  /= 255.

    y_train = y_train.astype(numpy.int32)
    y_test  = y_test.astype(numpy.int32)

    print()
    print('CIFAR-10 data loaded: train:',len(x_train),'test:',len(x_test))
    print('X_train:', x_train.shape)
    print('y_train:', y_train.shape)

# %% [markdown]
# This time we won't flatten the images upfront. 
# 
# The training data (`X_train`) is a 3rd-order tensor of size (50000, 32, 32), i.e. it consists of 50000 images of size 32x32 pixels. 
# 
# `y_train` is a 50000-dimensional vector containing the correct classes ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') for each training sample.

# %% [markdown]
# <img src="images/image_representation.png"  align="left"/>

# %% [markdown]
# ## Linear model
# 
# ### Initialization
# 
# Let's begin with a simple linear model, but with the Keras library. First we use a `Flatten` layer to convert image data into vectors. 
# 
# A `Dense()` layer is a basic layer: $xW + b$ with an optional nonlinearity applied ("activation function"). The `Dense` layer connects each input to each output with some weight parameter. They are also called "fully connected."
# 
# Here we add a `Dense` layer that has $32\times32\times3=3072$ input nodes (one for each pixel in the input image) and 10 output nodes. 

# %%
class LinearClassifier(tf.keras.models.Model):

    def __init__(self, activation=tf.nn.tanh):
        tf.keras.models.Model.__init__(self)

        self.layer_1 = tf.keras.layers.Dense(10, activation='softmax')


    def call(self, inputs):

        x = tf.keras.layers.Flatten()(inputs)
        x = self.layer_1(x)

        return x

# %% [markdown]
# We select *sparse categorical crossentropy* as the loss function, select [*stochastic gradient descent*](https://keras.io/optimizers/#sgd) as the optimizer, add *accuracy* to the list of metrics to be evaluated, and `compile()` the model. Note there are [several different options](https://keras.io/optimizers/) for the optimizer in Keras that we could use instead of *sgd*.

# %%
linear_model = LinearClassifier()

linear_model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])

# %% [markdown]
# ### Learning

# %% [markdown]
# Now we are ready to train our first model. An epoch means one pass through the whole training data.

# %% [markdown]
# Here is a concise way to train the network. The fit function handles looping over the batches. We'll see a more verbose approach in the next notebook that allows more performance tuning.

# %% [markdown]
# You can run the code below multiple times and it will continue the training process from where it left off. If you want to start from scratch, re-initialize the model using the code a few cells ago.

# %%
# This took about a third of a second per epoch on my laptop
batch_size = 512
epochs = 30
history = linear_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

# %%
print(linear_model.summary())

# %% [markdown]
# The summary shows that there are 30730 parameters in our model, as the weight matrix is of size 3072x10, plus there's a bias vector of 10x1.

# %% [markdown]
# Let's now see how the training progressed. 
# 
# * *Loss* is a function of the difference of the network output and the target values.  We are minimizing the loss function during training so it should decrease over time.
# * *Accuracy* is the classification accuracy for the training data (100*accuracy is the percentage labeled correctly), so it should increase over time
# 
# Note that for either measure, we cannot fully trust the progress, as the model may have overfitted and just memorized the training data.

# %%
plt.figure(figsize=(5,3))
plt.plot(history.epoch, history.history['loss'])
plt.title('loss')

plt.figure(figsize=(5,3))
plt.plot(history.epoch, history.history['accuracy'])
plt.title('accuracy')

# %% [markdown]
# ### Inference
# 
# For a better measure of the quality of the model, let's see the model accuracy for the test data. 

# %%
linscores = linear_model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (linear_model.metrics_names[1], linscores[1]*100))

# %% [markdown]
# We can now take a closer look on the results.
# 
# Let's define a helper function to show the failure cases of our classifier. 

# %%
def show_failures(predictions, trueclass=None, predictedclass=None, maxtoshow=20):
    rounded = numpy.argmax(predictions, axis=1)
    errors = rounded!=y_test.flatten()
    print('Showing max', maxtoshow, 'first failures. '
          'The predicted class is shown first and the correct class in parenthesis.')
    ii = 0
    plt.figure(figsize=(maxtoshow, 1))
    for i in range(x_test.shape[0]):
        if ii>=maxtoshow:
            break
        if errors[i]:
            if trueclass is not None and y_test[i] != trueclass:
                continue
            if predictedclass is not None and rounded[i] != predictedclass:
                continue
            plt.subplot(1, maxtoshow, ii+1)
            plt.axis('off')
            plt.imshow(x_test[i,:,:], cmap="gray")
            plt.title("%d (%d)" % (rounded[i], y_test[i]))
            ii = ii + 1

# %% [markdown]
# Here are the first 10 test images the linear model classified to a wrong class:

# %%
linpredictions = linear_model.predict(x_test)

show_failures(linpredictions)

# %% [markdown]
# ## Multi-layer perceptron (MLP) network
# 
# ### Initialization
# 
# Let's now create a more complex MLP model that has multiple layers, non-linear activation functions, and dropout layers. 
# 
# `Dropout()` randomly sets a fraction of inputs to zero during training, which is one approach to regularization and can sometimes help to prevent overfitting. 
# 
# There are two options below, a simple and a bit more complex model.  Select either one.
# 
# The output of the last layer needs to be a softmaxed 10-dimensional vector to match the groundtruth (`y_train`). 

# %%
class NonlinearClassifier(tf.keras.models.Model):

    def __init__(self, activation=tf.nn.tanh):
        tf.keras.models.Model.__init__(self)

        self.layer_1 = tf.keras.layers.Dense(50, activation='relu')
        
#         # A bit more complex model: (need to uncomment in call fn as well)
        self.layer_2 = tf.keras.layers.Dense(50, activation='relu')
        self.drop_3 = tf.keras.layers.Dropout(0.2)
        self.layer_4 = tf.keras.layers.Dense(50, activation='relu')
        self.drop_5 = tf.keras.layers.Dropout(0.2)
        
        # The last layer needs to be like this:
        self.layer_out = tf.keras.layers.Dense(10, activation='softmax')


    def call(self, inputs):

        x = tf.keras.layers.Flatten()(inputs)
        x = self.layer_1(x)
        
        # The more complex version:
        x = self.layer_2(x)
        x = self.drop_3(x)
        x = self.layer_4(x)
        x = self.drop_5(x)
        
        x = self.layer_out(x)

        return x

# %% [markdown]
# Finally, we again `compile()` the model, this time using [*RMSProp*](https://keras.io/optimizers/#rmsprop) as the optimizer.

# %%
nonlinear_model = NonlinearClassifier()

nonlinear_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# %% [markdown]
# ### Learning

# %%
# This took around half a second per epoch on my laptop for the simpler version, 
# and around 1 second per epoch for the more complex one.
batch_size = 512
epochs = 50
history = nonlinear_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

# %%
plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'])
plt.title('loss')

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['accuracy'])
plt.title('accuracy')

# %% [markdown]
# ### Inference
# 
# Accuracy for test data.  The model should be better than the linear model. 

# %%
scores = nonlinear_model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (nonlinear_model.metrics_names[1], scores[1]*100))

# %% [markdown]
# We can again take a closer look on the results, using the `show_failures()` function defined earlier.
# 
# Here are the first 10 test images the MLP classified to a wrong class:

# %%
predictions = nonlinear_model.predict(x_test)

show_failures(predictions)

# %% [markdown]
# We can use `show_failures()` to inspect failures in more detail. For example, here are failures in which the true class was "6":

# %%
show_failures(predictions, trueclass=6)

# %% [markdown]
# We can also compute the confusion matrix to see which image get mixed the most, and look at classification accuracies separately for each class:

# %%
from sklearn.metrics import confusion_matrix

print('Confusion matrix (rows: true classes; columns: predicted classes):'); print()
cm=confusion_matrix(y_test, numpy.argmax(predictions, axis=1), labels=list(range(10)))
print(cm); print()

j_sum = 0
print('Classification accuracy for each class:'); print()
for i,j in enumerate(cm.diagonal()/cm.sum(axis=1)): 
    print("%d: %.4f" % (i,j))
    j_sum += j
print('Average accuracy: %.4f %%' % (j_sum*10))

# %% [markdown]
# In the next notebook, we'll introduce convolutional layers, which are commonly used for images.

# %% [markdown]
# # In-class exercise: improve the accuracy of this model

# %% [markdown]
# How can you improve model accuracy by increasing epochs, stacking more layers, or changing the optimizer?

# %%
# Changing the optimzer to ADAM can help a lot, to 46%.


