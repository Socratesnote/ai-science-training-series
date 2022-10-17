# %% [markdown]
# # CIFAR-10 dataset classification with CNNs

# %% [markdown]
# # Homework: improve the accuracy of this model. Currently the model scores ~58% on the testing set.

# Changed:
# Batch size: 512 --> 128
# Epochs: 3 --> 100
# Learning rate: 0.1 --> 0.001
# Optimizer: ADAM --> RMSprop
# Model: base CIFAR10Classifier --> CIFAR10ClassifierAug (more filters and dense layers, no dropout).

# Validation accuracy 58% --> 65.8%. 

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

        # Fix to avoid invalid SSL certificates on Theta.
        import requests
        import ssl
        try:
            print("Downloading CIFAR10 with alternative SSL setup...")
            requests.packages.urllib3.disable_warnings()

            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                # Legacy Python that doesn't verify HTTPS certificates by default.
                pass
            else:
                # Handle target environment that doesn't support HTTPS verification.
                ssl._create_default_https_context = _create_unverified_https_context

            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        except:
            print("Downloading CIFAR10 with WGet...")
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

# CIFAR classifier with more filter layers, more dense layers, and no dropout.
class CIFAR10ClassifierAug(tf.keras.models.Model):

    def __init__(self, activation=tf.nn.tanh):
        tf.keras.models.Model.__init__(self)

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
def train_loop(dataset_train, dataset_test, batch_size, n_training_epochs, model, optimizer, silent = False):
    
    @tf.function()
    def train_iteration(batch_data, y_true, model, optimizer):
        # GradientTape keeps track of the gradients as they are calculated in the iterations. This lets you define a custom training loop.
        with tf.GradientTape() as tape:
            loss = forward_pass(model, batch_data, y_true)

        # In the Keras.io examples, they use trainable_weights?
        trainable_vars = model.trainable_variables

        # Apply the update to the network (one at a time):
        grads = tape.gradient(loss, trainable_vars)

        # Keras.io: note that this does not apply gradient clipping: you'd have to do that manually.
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss

    def validation_iteration(batch_data, y_true, model, optimizer):
        with tf.GradientTape() as tape:
            loss = forward_pass(model, batch_data, y_true)
        return loss

    # Initialize.
    avg_time = 0
    total_time = 0
    loss_train = numpy.zeros([n_training_epochs, 1])
    loss_test = numpy.zeros([n_training_epochs, 1])
    acc_train = numpy.zeros([n_training_epochs, 1])
    acc_test = numpy.zeros([n_training_epochs, 1])

    for i_epoch in range(n_training_epochs):
        start = time.time()
        if not silent:
            print("Epoch %d" % i_epoch)
        
        # Shuffle the whole dataset.
        dataset_train.shuffle(50000) 
        # Create a list of batches to iterate through.
        batches = dataset_train.batch(batch_size=batch_size, drop_remainder=True)
        loss_batch = numpy.zeros([len(batches), 1])
        acc_batch = numpy.zeros([len(batches), 1])
        for i_batch, (batch_data, y_true) in enumerate(batches):
            batch_data = tf.reshape(batch_data, [-1, 32, 32, 3])
            loss_batch[i_batch] = train_iteration(batch_data, y_true, model, optimizer)
            acc_batch[i_batch] = get_accuracy(model, batch_data, y_true, 10)[0]
        
        # Average loss across all batches.
        loss_train[i_epoch] = numpy.mean(loss_batch)
        # Get classification accuracy of training set.
        acc_train[i_epoch] = numpy.mean(acc_batch)

        # How do I "unslice" this dataset into the component x and y values?
        (x_test, y_test) = dataset_test.batch(batch_size=len(dataset_test),drop_remainder=True)
        # Get classification accuracy of validation set.
        acc_test[i_epoch] = get_accuracy(model, x_test, y_test, 10)[0]
        # Get loss of validation set.
        loss_test[i_epoch] = validation_iteration(x_test, y_test, model, optimizer)

        end = time.time()
        total_time += (end-start)
        avg_time = total_time / (i_epoch + 1)
        if not silent:
            print("Took %1.1f seconds for epoch #%d." % (end-start, i_epoch))

    print("Took %.1f s in total. (avg: %.3f / epoch)" % (total_time, avg_time))

    history = {'acc_train', acc_train, 'acc_test', acc_test, 'loss_train', loss_train, 'loss_test', loss_test}

    return history

# Training function.
def train_network(dataset_train, dataset_test, _model_type, _optimizer, _batch_size, _n_training_epochs, _lr, _silent = False):

    # Instantiate model.
    if _model_type == "base":
        mnist_model = CIFAR10Classifier()
    elif _model_type == "adhd":
        mnist_model = CIFAR10ClassifierAug()
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
    history = train_loop(dataset_train, dataset_test, _batch_size, _n_training_epochs, mnist_model, optimizer, _silent)

    return mnist_model, history

def get_accuracy(model, batch_data, batch_labels, n_classes):
    predictions = model.predict(batch_data)
    cm = confusion_matrix(batch_labels, numpy.argmax(predictions, axis=1), labels=list(range(n_classes)))

    j_sum = 0
    for i,j in enumerate(cm.diagonal()/cm.sum(axis=1)): 
        j_sum += j
    acc = 100*j_sum/n_classes
    return acc, cm
# %%
# Argument parser

# Default values.
batch_size = 128
epochs = 10
learning_rate = 0.001
model_type = "base"
optimizer_type = "adam"
silent = True
arg_help = "{0} -m <model_type> -o <optimizer_type> -b <batch_size> -e <epochs> -l <learning_rate> -s <silent>".format(sys.argv[0])

# Validate input.
try:
    opts, args = getopt.getopt(sys.argv[1:], "hm:o:b:e:l:s:", ["help", "model_type=", "optimizer_type=" "batch_size=", 
    "epochs=", "learning_rate=", "silent="])
except:
    print("Invalid input.")
    print(sys.argv)
    print(arg_help)
    opts = []

# Assign arguments.
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

# %%
# Create datasets.
dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# %%
# Train model.
print("Training model with hyperparameters:")
print("Model: %s. Optimizer: %s. BS: %i. Epochs: %i. LR: %f." % (model_type, 
optimizer_type, batch_size, epochs, learning_rate))

model, history = train_network(dataset_train, dataset_test, model_type, optimizer_type, batch_size, epochs, learning_rate, silent)

# %%
# Print confusion matrix and accuracy for the testing data.
acc, cm = get_accuracy(model, x_test, y_test, 10)
print()
print('Confusion matrix (rows: true classes; columns: predicted classes):')
print(cm)
print()

print("Model: %s. Optimizer: %s. Acc: %.1f. BS: %i. Epochs: %i. LR: %f." % (model_type, optimizer_type, acc, batch_size, epochs, learning_rate))

# Get datestring of now.
dt_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# %%
# Create accuracy and loss figures.
# plot loss
plt.subplot(211)
plt.plot(history["loss_train"], color='blue', label='train')
plt.plot(history["loss_test"], color='red', label='test')
plt.title('Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend(["Training", "Testing"])
# plot accuracy
plt.subplot(212)
plt.plot(history["acc_train"], color='blue', label='train')
plt.plot(history["acc_test"], color='red', label='test')
plt.title('Classification Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.legend(["Training", "Testing"])
# save plot to file
filename = sys.argv[0].split('/')[-1]
plt.savefig(dt_string + '_plot.png')
plt.close()

# %%
# Save model.
file_name = "%s_model_%s_opt_%s_acc_%.1f_BS_%i_LR_%.4f.tf" % (
    dt_string, model_type, optimizer_type, acc, batch_size, learning_rate)
model.save(file_name)
print("Model saved to '%s'." % (file_name))

# %%



