# %% [markdown]

# # Homework Session 2

# Homework: increase the accuracy of this loop.
# For example: change hyperparameters like epochs, learning rate, hidden-layer dimension, weight scale (standard deviation), etc.
# You're also allowed to go into the fc_net, and change the layer structure (e.g. add more layers)
# Currently is scores about 40% on the testing data, so just try to get higher. No worries about scoring 99%.
# To submit: just submit your version of the full_net notebook.

# %% 
# General imports.
import logging
import common_py.combilogger as clogger

# %%
# Configure combilogger.
log = clogger.Combilogger(logLevel=logging.DEBUG, logPath=".", fileName="", writeMode="w")
log.set_format_console('%(message)s')
log.set_format_file('%(asctime)s - %(levelname)s - %(message)s')
log.set_level_console(logging.INFO)
log.set_level_file(logging.DEBUG)
log.info("2022-09-27_ThomasPlaisier_Session02")

# %%
from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
    ipython.magic('matplotlib inline')

# %%
# Imports.
import tensorflow as tf

import numpy
import matplotlib.pyplot as plt

# %%
# Definitions.

# A simple implementation of stochastic gradient descent.
def sgd(model, gradients, learning_rate):
    for p, w in model.params.items():
        dw = gradients[p]
        new_weights = w - learning_rate * dw
        model.params[p] = new_weights
    return model

# One training step.
def learn(model, x_train, y_train_onehot, learning_rate):
    loss, gradients = model.loss(x_train, y_train_onehot)
    model = sgd(model, gradients, learning_rate)
    return loss, model

# Measure of accuracy.
def accuracy(model, x, true_values):
    scores = model.loss(x)
    predictions = numpy.argmax(scores, axis=1)
    N = predictions.shape[0]
    acc = (true_values == predictions).sum() / N
    return acc

# Model creator.
def model_builder(num_features, hidden_layer_width, nb_classes, initial_weight_scale):
    # The weights are initialized from a normal distribution with standard deviation (weight_scale).
    model = TwoLayerNet(input_dim=num_features, hidden_dim=hidden_layer_width, num_classes=nb_classes, weight_scale=initial_weight_scale)
    log.debug("Model created.")
    return model

# Here's an example training loop using this two-layer model. Can you do better? 
def model_trainer(model, x_train, y_train, y_train_onehot, num_epochs, batch_size, x_test, y_test):
    num_examples = x_train.shape[0]
    num_batches = int(num_examples / batch_size)
    # Pre-allocate.
    losses = numpy.zeros(num_batches*num_epochs,)
    indices = numpy.arange(num_examples)
    i = 0 # Iteration.

    for epoch in range(0, num_epochs):
        # In each epoch, we loop over all of the training examples.
        for step in range(0, num_batches):
            # Create this training batch.
            # Simply step through the entire set, and shuffle at the end.
            offset = step * batch_size
            batch_range = range(offset, offset+batch_size)
            x_train_batch = x_train[batch_range, :]
            y_train_batch = y_train_onehot[batch_range,:]
            
            # Perform one SGD step.
            loss, model = learn(model, x_train_batch, y_train_batch, learning_rate)
            losses[i] = loss
            i += 1

        acc = accuracy(model, x_train, y_train)
        log.debug("Epoch: %d. Loss: %.2f. Accuracy %.1f." % (epoch, loss, 100*acc))
        
        # Shuffle the data so that we get a new set of batches. Ensure that everything is shuffled the same way.
        numpy.random.shuffle(indices)
        x_train = x_train[indices,:]
        y_train = y_train[indices]
        y_train_onehot = y_train_onehot[indices,:]

    t_acc = accuracy(model, x_test, y_test)
    return t_acc, losses



# %%

# Load MNIST data from Keras.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
log.info('MNIST data loaded. # Training elements: %i, # Testing elements: %i.' % (len(x_train), len(x_test)))

# %%
# Data preparation

# Convert from uint8 to float32.
x_train = x_train.astype(numpy.float32)
x_test  = x_test.astype(numpy.float32)

# Normalize range from [0, 255] to [0, 1].
x_train /= 255.
x_test  /= 255.

log.info('Input data is of shape: %s' % (list(x_train.shape)))
# Reshape data from [N, d1, d2] to [N, d1*d2].
x_train = x_train.reshape(x_train.shape[0], numpy.prod(x_train[0,:,:].shape))
x_test = x_test.reshape(x_test.shape[0], numpy.prod(x_test[0,:,:].shape))
log.info('Input data reshaped to: %s' % (list(x_train.shape)))

# Convert from uint8 to int32.
y_train = y_train.astype(numpy.int32)
y_test  = y_test.astype(numpy.int32)
log.info('Label data is of shape: %s' % (list(y_train.shape)))

# Convert from real integers to one-hot encoding (categorical):
num_features = x_train.shape[1] # Number of pixels
nb_classes = 10
y_train_onehot = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, nb_classes)

# %%
# Import neural network.
# This implementation is a two-layer neural network. Credit: Stanford's CSE231n course, 
# hosted at https://github.com/cs231n/cs231n.github.io with the MIT license.
from fc_net import TwoLayerNet

# %%
# Hyperparameters
# Number of units in hidden layers.
hidden_layer_width = 300
# Standard divation of normal distribution of weights.
initial_weight_scale = 0.01
# Learning rate of model.
learning_rate = 0.01  
# Batch size for SGD.
batch_size = 10000
# Number of epochs.
# num_epochs = 10
# num_epochs_range = numpy.round(numpy.linspace(5, 50, 5))
num_epochs_range = numpy.round(numpy.linspace(5, 50, 6)).astype(numpy.int32)

# %%
# Let's start simple, and modulate number of epochs.
for num_epochs in num_epochs_range:
    model = model_builder(num_features, hidden_layer_width, nb_classes, initial_weight_scale)

    t_acc, losses = model_trainer(model, x_train, y_train, y_train_onehot, num_epochs, batch_size, x_test, y_test)

    # Plot loss trajectory.
    # plt.title('Loss trajectory')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.plot(losses)

    # Get accuracy from test.
    log.info("Hyperparameters. HLW: %i. WS: %.3f. LR: %.3f. Batch: %.0f. Epochs: %i." % (hidden_layer_width, initial_weight_scale, learning_rate, batch_size, num_epochs))
    log.info("Testing accuracy: %.1f." % (100*t_acc))

# %% 
# Show predictions for 10 random images.
indices = numpy.random.randint(0, x_train.shape[0], 10)
x_disp = x_train[indices,:]
scores = model.loss(x_disp)
predictions = numpy.argmax(scores, axis=1)

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(numpy.reshape(x_disp[i,:], (28, 28)), cmap="gray")
    plt.title('%.0f' % predictions[i])

# %%

# Now to optimize things, let's do a grid search.
# We'll be tuning 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

width_hidden_layer_range = numpy.round(numpy.linspace(50, 500, 5))
initial_weight_scale_range = numpy.linspace(0.005, 0.100, 5)
# Learning rate of model.
learning_rate_range = numpy.linspace(0.001, 0.02, 5)
# Batch size for SGD.
batch_size_range = numpy.round(numpy.linspace(1000, 20000, 5))
# Number of epochs.
num_epochs_range = numpy.round(numpy.linspace(5,100, 5))

param_grid = dict(width_hidden_layer=width_hidden_layer_range, initial_weight_scale=initial_weight_scale_range, learning_rate=learning_rate_range, batch_size = batch_size_range, num_epochs = num_epochs_range)

# %%
# Define custom scorer.
def my_custom_score_func(model, x_test, y_test):
     acc = accuracy(model, x_test, y_test)
     return acc 

score = make_scorer(my_custom_score_func, greater_is_better=True)

grid = GridSearchCV(estimator = model, scoring = score, param_grid=param_grid, cv=5) # 5 iterations.

# %%

grid.fit(x_train, y_train)

# %%
log.info("Best parameters: %s. Best score: %.2f." % (grid.best_params_, grid.best_score_))
