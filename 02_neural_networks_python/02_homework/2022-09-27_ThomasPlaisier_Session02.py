# %% [markdown]

# # Homework Session 2

# Homework: increase the accuracy of this loop.
# For example: change hyperparameters like epochs, learning rate, hidden-layer dimension, weight scale (standard deviation), etc.
# You're also allowed to go into the fc_net, and change the layer structure (e.g. add more layers)
# Currently is scores about 40% on the testing data, so just try to get higher. No worries about scoring 99%.
# To submit: just submit your version of the full_net notebook.

# %%
# General imports.
from fc_net import TwoLayerNet
import ssl
import requests
import pickle
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from IPython import get_ipython
import logging
import common_py.combilogger as clogger
import datetime
import time
from sklearn.model_selection import ParameterGrid

# %%
# Configure combilogger.
dt_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

log = clogger.Combilogger(logLevel=logging.DEBUG,
                          logPath=".", fileName=dt_string, writeMode="w")
log.set_format_console('%(message)s')
log.set_format_file('%(asctime)s - %(levelname)s - %(message)s')
log.set_level_console(logging.INFO)
log.set_level_file(logging.DEBUG)
log.info("2022-09-27_ThomasPlaisier_Session02")

# %%
# Special settings for running in interactive mode.
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
    ipython.magic('matplotlib inline')

# %%
# Imports.

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
    model = TwoLayerNet(input_dim=num_features, hidden_dim=hidden_layer_width,
                        num_classes=nb_classes, weight_scale=initial_weight_scale)
    log.debug("Model created.")
    return model

# Model trainer.


def model_trainer(model, x_train, y_train, y_train_onehot, max_epochs, batch_size, x_test, y_test):
    num_examples = x_train.shape[0]
    num_batches = int(num_examples / batch_size)
    # Pre-allocate.
    losses = numpy.zeros(num_batches*max_epochs,)
    indices = numpy.arange(num_examples)
    i = 0  # Iteration.
    acc_prev = 0
    delta_acc = 5*10**-3 # Stop when gain is less than 0.5 %.

    start_time = time.time()

    for epoch in range(0, max_epochs):
        # In each epoch, we loop over all of the training examples.
        for step in range(0, num_batches):
            # Create this training batch.
            # Simply step through the entire set, and shuffle at the end.
            offset = step * batch_size
            batch_range = range(offset, offset+batch_size)
            x_train_batch = x_train[batch_range, :]
            y_train_batch = y_train_onehot[batch_range, :]

            # Perform one SGD step.
            loss, model = learn(model, x_train_batch,
                                y_train_batch, learning_rate)
            losses[i] = loss
            i += 1

        # After stepping through the entire set, determine accuracy.
        acc = accuracy(model, x_train, y_train)
        log.debug("Epoch: %d. Loss: %.2f. Accuracy %.1f." %
                  (epoch, loss, 100*acc))

        # When accuracy stalls, stop the training loop: no point in performing 50 epochs when 7 will get you most of the way there.
        if (numpy.abs(acc - acc_prev) < delta_acc) & (epoch > 5):
            log.info("Breaking at epoch %d of %d due to stalling accuracy gains (|%.1f| < %.1f)." % (
                epoch, max_epochs, 100*(acc - acc_prev), 100*delta_acc))
            break

        # Store.
        acc_prev = acc

        # Shuffle the data so that we get a new set of batches. Ensure that everything is shuffled the same way.
        numpy.random.shuffle(indices)
        x_train = x_train[indices, :]
        y_train = y_train[indices]
        y_train_onehot = y_train_onehot[indices, :]

    end_time = time.time()
    log.info('Model took %.1f s to train.' % (end_time - start_time))

    # After model is trained, determine accuracy with the testing dataset.
    t_acc = accuracy(model, x_test, y_test)
    return t_acc, losses

# Save model to file.


def save_model(model, t_acc):
    dt_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    params = model.get_params()
    file_name = "model_acc_%.1f_HLW_%i_WS_%.3f_LR_%.3f_BS_%.0f_ME_%i_%s.pkl" % (
        100*t_acc, hidden_layer_width, initial_weight_scale, learning_rate, batch_size, max_epochs, dt_string)
    file_handle = open(file_name, "wb")
    pickle.dump(params, file_handle)
    file_handle.close()
    log.info("Model saved to '%s'." % (file_name))


# %%
# Fix to avoid invalid SSL certificates on Theta.
requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default.
    pass
else:
    # Handle target environment that doesn't support HTTPS verification.
    ssl._create_default_https_context = _create_unverified_https_context

# %%

# Load MNIST data from Keras.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
log.info('MNIST data loaded. # Training elements: %i, # Testing elements: %i.' %
         (len(x_train), len(x_test)))

# %%
# Data preparation

# Convert from uint8 to float32.
x_train = x_train.astype(numpy.float32)
x_test = x_test.astype(numpy.float32)

# Normalize range from [0, 255] to [0, 1].
x_train /= 255.
x_test /= 255.

log.info('Input data is of shape: %s' % (list(x_train.shape)))
# Reshape data from [N, d1, d2] to [N, d1*d2].
x_train = x_train.reshape(x_train.shape[0], numpy.prod(x_train[0, :, :].shape))
x_test = x_test.reshape(x_test.shape[0], numpy.prod(x_test[0, :, :].shape))
log.info('Input data reshaped to: %s' % (list(x_train.shape)))

# Convert from uint8 to int32.
y_train = y_train.astype(numpy.int32)
y_test = y_test.astype(numpy.int32)
log.info('Label data is of shape: %s' % (list(y_train.shape)))

# Convert from real integers to one-hot encoding (categorical):
num_features = x_train.shape[1]  # Number of pixels
nb_classes = 10
y_train_onehot = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, nb_classes)

# %%
# Import neural network.
# This implementation is a two-layer neural network. Credit: Stanford's CSE231n course,
# hosted at https://github.com/cs231n/cs231n.github.io with the MIT license.

# %%
# Hyperparameters
# Number of units in hidden layers.
hidden_layer_width = 300  # Default, 98.2
# hidden_layer_width = 3000 # 98.2, but takes ages.
# hidden_layer_width = 30 # 96.5
# hlw_range = [3, 30, 100, 300, 1000]
hlw_range = [3, 30]

# Standard divation of normal distribution of weights.
initial_weight_scale = 0.01  # Default, 98.2
# initial_weight_scale = 0.001 # 97.9
# initial_weight_scale = 0.1 # 97.8
# initial_weight_scale = 1.0 # 93.4
# iws_range = [0.001, 0.005, 0.01, 0.1, 1.0]
iws_range = [0.01, 0.05]

# Learning rate of model.
# learning_rate = 5.0 # 10.3
# learning_rate = 1.0 # 97.7
# learning_rate = 0.7 # 98
learning_rate = 0.5  # 98.2
# learning_rate = 0.2 # 97.5
# learning_rate = 0.1 # % 96.6
# learning_rate = 0.06 # 95.4%
# learning_rate = 0.04 # 94.2%
# learning_rate = 0.01 # Default
# lr_range = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
lr_range = [0.1, 0.5]

# Batch size for SGD.
batch_size = 100  # Pretty good, 90% with the rest default.
# batch_size = 10000 # Default
# bs_range = [100, 1000, 5000, 10000, 30000]
bs_range = [100, 10000]

# Number of epochs.
# In most cases you don't need to go higher than 30. The loop will automatically break when stalling.
max_epochs = 50
# num_epochs_range = numpy.round(numpy.linspace(15, 52, 6)).astype(numpy.int32)

# %%
# Build a parameter grid.
param_grid = {'HLW': hlw_range, 'IWS': iws_range,
              'LR': lr_range, 'BS': bs_range}

# %%
def run_set(set):
    hidden_layer_width = set['HLW']
    initial_weight_scale = set['IWS']
    learning_rate = set['LR']
    batch_size = set['BS']

    log.info("Hyperparameters set %i of %i. HLW: %i. WS: %.3f. LR: %.3f. BS: %.0f. ME: %i." % (
        i, i_max, hidden_layer_width, initial_weight_scale, learning_rate, batch_size, max_epochs))
    
    model = model_builder(num_features, hidden_layer_width,
                        nb_classes, initial_weight_scale)

    t_acc, losses = model_trainer(
        model, x_train, y_train, y_train_onehot, max_epochs, batch_size, x_test, y_test)
    log.info("Testing accuracy: %.1f." % (100*t_acc))

    save_model(model, t_acc)

# %%
# Build, train, and test model across all sets in the parameter grid.
grid_list = list(ParameterGrid(param_grid))
i = -1
i_max = len(grid_list)
# Store all accuracies to find out which set was best.
t_acc_all = numpy.zeros(i_max)

# %%
t2 = time.time()

# Parallel
import multiprocessing
if __name__ == '__main__': # Only main thread can spawn these.
    with multiprocessing.Pool() as pool:
        t_acc_all = pool.map(run_set, grid_list)

log.info('Parallel took %.1f s.' % (time.time() - t2))

# %%

t1 = time.time()
# Sequential.
for set in grid_list:
    i = grid_list.index(set)
    t_acc = run_set(set)
    t_acc_all[i] = t_acc

log.info('Sequential took %.1f s.' % (time.time() - t1))


# %%
# Show best model.
log.warning("Best model:")
best_acc_index = numpy.argmax(t_acc_all)
log.warning("Testing accuracy: %.1f" % (100*t_acc_all[best_acc_index]))
best_set = ParameterGrid(param_grid)[best_acc_index]
log.warning("Model: %s" % (best_set))

# %%
# # Show predictions for 10 random images for last model.
# indices = numpy.random.randint(0, x_train.shape[0], 10)
# x_disp = x_train[indices, :]
# scores = model.loss(x_disp)
# predictions = numpy.argmax(scores, axis=1)

# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     plt.axis('off')
#     plt.imshow(numpy.reshape(x_disp[i, :], (28, 28)), cmap="gray")
#     plt.title('%.0f' % predictions[i])

# %%
