# %% [markdown]
# # CIFAR-10 dataset classification with CNNs

# %% [markdown]
# # Homework: improve the accuracy of this model. Currently the model scores ~58% on the testing set.

# Changed:
# Batch size: 512 --> 128
# Epochs: 3 --> 100
# Learning rate: 0.1 --> 0.01
# Accuracy 58% --> 65%. 

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
import datetime

# Fixes issue with Tensorflow crashing?
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Image loader for importing from folder.
try:
    from image_dataset_loader import load
except:
    os.system("https_proxy=http://proxy.tmi.alcf.anl.gov:3128  pip install image-dataset-loader")
    from image_dataset_loader import load

# %%

# If the dataset has already been pre-processed, just load it from a stored Numpy array.
file = glob.glob('**/cifar_store.npz', recursive=True) # Search everywhere, because debugging launches the file from $HOME whereas qsub runs it from the homework folder.
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
    
    # After loading raw data from folder, shape.
    x_train = x_train.astype(numpy.float32)
    x_test  = x_test.astype(numpy.float32)

    x_train /= 255.
    x_test  /= 255.

    y_train = y_train.astype(numpy.int32)
    y_test  = y_test.astype(numpy.int32)
    # Save shaped data to file.
    with open("cifar_store.npz", "wb") as f:
        numpy.savez(f, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

# %%
# Definitions

# CIFAR10 Classifier class.
class CIFAR10Classifier(tf.keras.models.Model):

    def __init__(self, activation=tf.nn.tanh):
        tf.keras.models.Model.__init__(self)

        # Filter layer: 32 3x3 kernels.
        self.conv_1 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')
        # Filter layer: 64 3x3 kernels.
        self.conv_2 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu')
        # Subsampling layer: max pooling in 2x2.
        self.pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # Regularization.
        self.drop_4 = tf.keras.layers.Dropout(0.25)
        # Dense 128 output layer.
        self.dense_5 = tf.keras.layers.Dense(128, activation='relu')
        # Regularization.
        self.drop_6 = tf.keras.layers.Dropout(0.5)
        # Dense 10 class output layer.
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

# CIFAR classifier based on https://github.com/adhishthite/cifar10-optimizers and https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
class CIFAR10ClassifierADHD(tf.keras.models.Model):

    def __init__(self, activation=tf.nn.tanh):
        tf.keras.models.Model.__init__(self)

        # Useful insight from MLM: typically, when stacking convolutional layers, you start with e.g. 32 in the first set, and then double the kernel size for each additional set.
        # Filter layer: 32 3x3 kernels.
        self.conv_1 = tf.keras.layers.Conv2D(32, [3, 3], padding="same", activation='relu')
        # Filter layer: 32 3x3 kernels.
        self.conv_2 = tf.keras.layers.Conv2D(32, [3, 3], padding="same", activation='relu')
        # Subsampling layer: max pooling in 2x2.
        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # Filter layer: 64 3x3 kernels.
        self.conv_3 = tf.keras.layers.Conv2D(64, [3, 3], padding="same", activation='relu')
        # Filter layer: 64 3x3 kernels.
        self.conv_4 = tf.keras.layers.Conv2D(64, [3, 3], padding="same", activation='relu')
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # Filter layer: 128 3x3 kernels.
        self.conv_5 = tf.keras.layers.Conv2D(128, [3, 3], padding="same", activation='relu')
        # Filter layer: 128 3x3 kernels.
        self.conv_6 = tf.keras.layers.Conv2D(128, [3, 3], padding="same", activation='relu')
        # Subsampling layer: max pooling in 2x2.
        self.pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # Dense 512 output layer.
        self.dense_1 = tf.keras.layers.Dense(512, activation='relu')
        # Dense 128 output layer.
        self.dense_2 = tf.keras.layers.Dense(128, activation='relu')
        # Dense 32 output layer.
        self.dense_3 = tf.keras.layers.Dense(32, activation='relu')
        # Dense 10 output layer.
        self.dense_4 = tf.keras.layers.Dense(10, activation='relu')
        # Dense 10 class output layer.
        self.dense_class = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):

        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.pool_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.pool_2(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.pool_3(x)
        # Flatten because we will be feeding to dense layers next.
        x = tf.keras.layers.Flatten()(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_class(x)

        return x

# Loss function of model.
def compute_loss(y_true, y_pred):
    # If labels were one-hot encoded, use standard crossentropy.
    # Since labels are integers, use sparse categorical cross-entropy. 
    # The network's final layer is softmax, so from_logtis=False
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    return scce(y_true, y_pred)  

# Forward pass of model.
def forward_pass(model, batch_data, y_true):
    y_pred = model(batch_data)
    loss = compute_loss(y_true, y_pred)

    return loss

# Training loop manager.
def train_loop(dataset, batch_size, n_training_epochs, model, optimizer, silent = False):
    
    # Decorate the training iteration. Is this needed? --> Yes it is, otherwise the training loop can't calculate gradients properly. That is, it throws a whole bunch of warnings.
    @tf.function()
    def train_iteration(data, y_true, model, optimizer):
        # GradientTape keeps track of the gradients as they are calculated in the iterations. This lets you define a custom training loop.
        with tf.GradientTape() as tape:
            loss = forward_pass(model, data, y_true)

        # In the Keras.io examples, they use trainable_weights?
        trainable_vars = model.trainable_variables

        # Apply the update to the network (one at a time):
        grads = tape.gradient(loss, trainable_vars)

        # Keras.io: note that this does not apply gradient clipping: you'd have to do that manually.
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss

    # Initialize.
    avg_time = 0
    total_time = 0
    for i_epoch in range(n_training_epochs):
        start = time.time()
        if not silent:
            print("Epoch %d" % i_epoch)
        
        # Shuffle the whole dataset.
        dataset.shuffle(50000) 
        # Create a list of batches to iterate through.
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
def train_network(dataset, _model_type, _optimizer, _batch_size, _n_training_epochs, _lr, _silent = False):

    # Instantiate model.
    if _model_type == "base":
        mnist_model = CIFAR10Classifier()
    elif _model_type == "adhd":
        mnist_model = CIFAR10ClassifierADHD()
    else:
        mnist_model = CIFAR10Classifier()

    # Define optimizer.
    if _optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=_lr)
    if _optimizer == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=_lr)
    elif _optimizer == "sgd":
        mnist_model = tf.keras.optimizers.SGD(learning_rate=_lr)
    else:
        optimizer = tf.keras.optimizers.Adam(_lr)

    # Train model with given hyperparameters.
    train_loop(dataset, _batch_size, _n_training_epochs, mnist_model, optimizer, _silent)
    return mnist_model


# %%
# Hyperparameters.
# %%
# Argument parser
batch_size = 128
epochs = 100
learning_rate = 0.001
model_type = "base"
optimizer_type = "adam"
silent = True
arg_help = "{0} -m <model_type> -o <optimizer_type> -b <batch_size> -e <epochs> -l <learning_rate> -s <silent>".format(sys.argv[0])

try:
    opts, args = getopt.getopt(sys.argv[1:], "hm:o:b:e:l:s:", ["help", "model_type=", "optimizer_type=" "batch_size=", 
    "epochs=", "learning_rate=", "silent="])
except:
    print("Invalid input.")
    print(sys.argv)
    print(arg_help)
    opts = []

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print(arg_help)
        sys.exit()
    elif opt in ("-m", "--model_type"):
        model_type = arg.lower()
    elif opt in ("-o", "--optimizer_type"):
        optimizer_type = arg.lower()
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
# BS: 128. Epochs: 100. LR: 0.000100. Avg accuracy: 65.4500 %.

# %%
# Train model on training data.
print("Training model with hyperparameters:")
print("Model: %s. Optimizer: %s. BS: %i. Epochs: %i. LR: %f." % (model_type, optimizer_type, batch_size, epochs, learning_rate))
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
model = train_network(dataset, model_type, optimizer_type, batch_size, epochs, learning_rate, silent)

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
print("Model: %s. Optimizer: %s. BS: %i. Epochs: %i. LR: %f." % (model_type, optimizer_type, batch_size, epochs, learning_rate))

# %%
# Save model.
dt_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_name = "model_%s_opt_%s_acc_%.1f_BS_%i_LR_%i.tf" % (
    model_type, optimizer_type, j_sum*10, batch_size, learning_rate)
model.save(file_name)
print("Model saved to '%s'." % (file_name))

# %% [markdown]
# Update this notebook to ensure more accuracy. How high can it be raised? Changes like increasing the number of epochs, altering the learning weight, altering the number of neurons the hidden layer, changing the optimizer, etc. could be made directly in the notebook. You can also change the model specification by expanding the network's layer. The current notebook's training accuracy is roughly 58.69%, although it varies randomly.

# %%



