# %% [markdown]
# # CIFAR-10 dataset classification with CNNs

# %% [markdown]
# # Homework: improve the accuracy of this model. Currently the model scores ~58% on the testing set.

# %%
# Imports.
import tensorflow as tf

import numpy
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import glob
import getopt
import sys

# Fixes issue with Tensorflow crashing?
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Image loader for importing from folder.
os.system("https_proxy=http://proxy.tmi.alcf.anl.gov:3128  pip install image-dataset-loader")
from image_dataset_loader import load

# %%

# If the dataset has already been pre-processed, just load it from a stored Numpy array.
file = glob.glob('./cifar_store.npz')
if len(file) != 0:
    print("Found pre-shaped data. Loading from Numpy...")
    this_file = numpy.load(file[0])
    x_train = this_file["x_train"]
    x_test = this_file["x_test"]
    y_train = this_file["y_train"]
    y_test = this_file["y_test"]
    print("Data loaded.")
else:
    # No stored Numpy. Check if the cifar folder exists.
    folder = glob.glob('./cifar10')
    if len(folder) == 0:
        # Folder does not exist. Download from source.
        print("Downloading CIFAR10.")
        os.system("pip install wget")
        import wget
        os.system("https_proxy=http://proxy.tmi.alcf.anl.gov:3128  wget https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz")
        os.system("tar -xf cifar10.tgz")
    else:
        print("Folder found. Loading with image_dataset_loader...")
        # Use image-dataset-loader to import the data. If this doesn't work, restart the kernel to refresh the package. If it does work, it takes _ages_.
        (x_train, y_train), (x_test, y_test) = load('cifar10', ['train', 'test'])
        print("Data loaded.")
    # Shape data.
    x_train = x_train.astype(numpy.float32)
    x_test  = x_test.astype(numpy.float32)

    x_train /= 255.
    x_test  /= 255.

    y_train = y_train.astype(numpy.int32)
    y_test  = y_test.astype(numpy.int32)
    # Save to file.
    with open("cifar_store.npz", "wb") as f:
        numpy.savez(f, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

# %%
# Definitions

# CIFAR10 Classifier class.
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

# Concise training function.
def train_network_concise(_batch_size, _n_training_epochs, _lr):

    cnn_model = CIFAR10Classifier()

    cnn_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    history = cnn_model.fit(x_train, y_train, batch_size=_batch_size, epochs=_n_training_epochs)
    return history, cnn_model

# Loss function of model.
def compute_loss(y_true, y_pred):
    # If labels are integers, use sparse categorical cross-entropy. 
    # The network's final layer is softmax, so from_logtis=False
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # If labels were one-hot encoded, use standard crossentropy
    return scce(y_true, y_pred)  

# Forward pass of model.
def forward_pass(model, batch_data, y_true):
    y_pred = model(batch_data)
    loss = compute_loss(y_true, y_pred)
    return loss

# Training loop manager.
def train_loop(dataset, batch_size, n_training_epochs, model, optimizer, silent = False):
    
    @tf.function()
    def train_iteration(data, y_true, model, opt):
        # What is tape?
        with tf.GradientTape() as tape:
            loss = forward_pass(model, data, y_true)

        trainable_vars = model.trainable_variables

        # Apply the update to the network (one at a time):
        grads = tape.gradient(loss, trainable_vars)

        opt.apply_gradients(zip(grads, trainable_vars))
        return loss

    avg_time = 0
    total_time = 0
    for i_epoch in range(n_training_epochs):
        if not silent:
            print("Epoch %d" % i_epoch)
        start = time.time()

        epoch_steps = int(50000/batch_size)
        dataset.shuffle(50000) # Shuffle the whole dataset in memory
        batches = dataset.batch(batch_size=batch_size, drop_remainder=True)
        
        for i_batch, (batch_data, y_true) in enumerate(batches):
            batch_data = tf.reshape(batch_data, [-1, 32, 32, 3])
            loss = train_iteration(batch_data, y_true, model, optimizer)
            
        end = time.time()
        total_time += (end-start)
        avg_time = total_time / (i_epoch + 1)
        if not silent:
            print("Took %1.1f seconds for epoch #%d." % (end-start, i_epoch))

    print("Took %.1f s in total. (avg: %.3f / epoch)" % (total_time, avg_time))

# Training function.
def train_network(dataset, _batch_size, _n_training_epochs, _lr, _silent = False):

    mnist_model = CIFAR10Classifier()

    optimizer = tf.keras.optimizers.Adam(_lr)

    train_loop(dataset, _batch_size, _n_training_epochs, mnist_model, optimizer, _silent)
    return mnist_model


# %%
# Hyperparameters.
# %%
# Argument parser
batch_size = 128
epochs = 100
learning_rate = .0001
silent = True
arg_help = "{0} -b <batch_size> -e <epochs> -l <learning_rate> -s <silent>".format(sys.argv[0])

try:
    opts, args = getopt.getopt(sys.argv[1:], "hb:e:l:s:", ["help", "batch_size=", 
    "epochs=", "learning_rate=", "silent="])
except:
    print("Invalid input.")
    print(sys.argv)
    print(arg_help)
    sys.exit()

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print(arg_help)
        sys.exit()
    elif opt in ("-b", "--batch_size"):
        batch_size = int(arg)
    elif opt in ("-e", "--epochs"):
        epochs = int(arg)
    elif opt in ("-l", "--learning_rate"):
        learning_rate = float(arg)
    elif opt in ("-s", "--silent"):
        silent = arg.lower() == 'true'

# %% [markdown]

# BS: 128. Epochs: 100. LR: 0.001 = 65.8400 %
# BS: 1024. Epochs: 100. LR: 0.01 = 50.9800 %
# BS: 128. Epochs: 500. LR: 0.001. Avg accuracy: 65.4300 %.

# %%
# Train model on training data.
print("Training model with hyperparameters:")
print("BS: %i. Epochs: %i. LR: %f." % (batch_size, epochs, learning_rate))
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset.shuffle(50000)
model = train_network(dataset, batch_size, epochs, learning_rate, silent)

# %%
# Print confusion matrix and accuracy for the testing data.
print('Confusion matrix (rows: true classes; columns: predicted classes):')
predictions = model.predict(x_test)
cm=confusion_matrix(y_test, numpy.argmax(predictions, axis=1), labels=list(range(10)))
print(cm)
print()

j_sum = 0
print('Classification accuracy for each class:')
for i,j in enumerate(cm.diagonal()/cm.sum(axis=1)): 
    print("%d: %.4f" % (i,j))
    j_sum += j
print("BS: %i. Epochs: %i. LR: %f. Avg accuracy: %.4f %%." % (batch_size, epochs, learning_rate, j_sum*10))

# %% [markdown]
# Update this notebook to ensure more accuracy. How high can it be raised? Changes like increasing the number of epochs, altering the learning weight, altering the number of neurons the hidden layer, changing the optimizer, etc. could be made directly in the notebook. You can also change the model specification by expanding the network's layer. The current notebook's training accuracy is roughly 58.69%, although it varies randomly.

# %%



