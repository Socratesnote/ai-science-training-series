
# %%
import sys, os
import time

# This limits the amount of memory used:
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import tensorflow as tf
import glob
import numpy
import matplotlib.pyplot as plt
import datetime
import getopt

# %%

# Residual layer.
class ResidualLayer(tf.keras.Model):

    def __init__(self, n_filters):
        # tf.keras.Model.__init__(self)
        super(ResidualLayer, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )

        self.norm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )

        self.norm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        
        # Note that this is the same step as the self.identity stage of the residual downsample layer, but with kernel size (1,1) and stride (1,1) --> simple equality.
        x = inputs

        output1 = self.norm1(self.conv1(inputs))

        output1 = tf.keras.activations.relu(output1)

        output2 = self.norm2(self.conv2(output1))

        # Note the residual here as "+ x".
        return tf.keras.activations.relu(output2 + x)

# Downsampling layer.
class ResidualDownsample(tf.keras.Model):

    # These correspond to the dashed lines in the Residual Layer paper in the 34 layer model.
    def __init__(self, n_filters):
        super(ResidualDownsample, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same",
            strides     = (2,2)
        )

        self.identity = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (1,1),
            strides     = (2,2),
            padding     = "same"
        )

        self.norm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )

        self.norm2 = tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, inputs):

        x = self.identity(inputs)
        output1 = self.norm1(self.conv1(inputs))
        output1 = tf.keras.activations.relu(output1)

        output2 = self.norm2(self.conv2(output1))

        return tf.keras.activations.relu(output2 + x)

# %%

# Define Resnet.
class ResNet34(tf.keras.Model):

    def __init__(self):
        super(ResNet34, self).__init__()

        # Sequence of downsampling with conv2D, normalization, and maxpool.
        self.conv_init = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters     = 64,
                kernel_size = (7,7),
                strides     = (2,2),
                padding     = "same",
                use_bias    = False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")
        ])

        # Sequence of 1:1 residuals.
        self.residual_series_1 = tf.keras.Sequential([
            ResidualLayer(64),
            ResidualLayer(64),
            ResidualLayer(64),
        ])

        # Increase the number of filters:
        self.downsample_1 = ResidualDownsample(128)

        # Sequence of 1:1 residuals.
        self.residual_series_2 = tf.keras.Sequential([
            ResidualLayer(128),
            ResidualLayer(128),
            ResidualLayer(128),
        ])

        # Increase the number of filters:
        self.downsample_2 = ResidualDownsample(256)

        # Sequence of 1:1 residuals.
        self.residual_series_3 = tf.keras.Sequential([
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        ])

        # Increase the number of filters:
        self.downsample_3 = ResidualDownsample(512)

        # Once more unto the breach for 1:1.
        self.residual_series_4 = tf.keras.Sequential([
            ResidualLayer(512),
            ResidualLayer(512),
        ])

        # Smoothing with an averaging layer.
        self.final_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(8,8)
        )

        # Finally, flatten and classify with a dense layer.
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(1000)

    @tf.function
    def call(self, inputs):

        x = self.conv_init(inputs)

        x = self.residual_series_1(x)
        x = self.downsample_1(x)

        x = self.residual_series_2(x)
        x = self.downsample_2(x)

        x = self.residual_series_3(x)
        x = self.downsample_3(x)

        x = self.residual_series_4(x)
        x = self.final_pool(x)

        x = self.flatten(x)

        logits = self.classifier(x)
        
        # Return results as logits (probabilities).
        return logits


# %%

# Define helper functions.

# Define a search function for the .json configuration. This allows the net to be trained both as a batch submission (which uses the local folder by definition) and the interactive mode (which can be anywhere).
def get_config_file():
    print('Looking for ilsvrc.json.')
    json_files = glob.glob("**/ilsvrc.json", recursive=True) 
    if len(json_files) > 1:
        print('Found candidates:')
        print(json_files)
        print('Filtering...')
        this_file = list(filter(lambda element: '04_homework' in element, json_files))
        if len(this_file) > 1:
            print('Error: more than 1 match for filter.')
            print(this_file)
            sys.exit()
    else:
        print('Found 1 file:')
        this_file = json_files
        print(this_file)

    this_file = this_file[0]
    print('Using file:')
    print(this_file)
    return this_file

# Define a data loader.
def prepare_data_loader(BATCH_SIZE):

    # Define number of threads to use for parallel loading: at most 8.
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    print("Parameters set, preparing dataloading")
    import json
    from ilsvrc_dataset import get_datasets

    # What is this supposed to fake?
    class FakeHvd:

        def size(self): return 1

        def rank(self): return 0

    # Add this to find the ilsvrc file from session 4.
    this_file = get_config_file()

    with open(this_file, 'r') as f:
        config = json.load(f)

    print(json.dumps(config, indent=4))

    config['hvd'] = FakeHvd()
    config['data']['batch_size'] = BATCH_SIZE

    train_ds, val_ds = get_datasets(config)
    print("Datasets ready, creating network.")

    return train_ds, val_ds

# Accuracy of logits versus true labels.
@tf.function
def calculate_accuracy(logits, labels):
    # We calculate top1 accuracy (one-hot encoding) only here:
    selected_class = tf.argmax(logits, axis=1)

    # Comparing this as floats seems a bit weird, but OK.
    correct = tf.cast(selected_class, tf.float32) == tf.cast(labels, tf.float32)

    # reduce_mean just calculates the mean of a tensor.
    mean_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return mean_accuracy

# Loss of logits versus labels.
@tf.function
def calculate_loss(logits, labels):
    # Note that _with_logits is chosen, otherwise the network won't converge.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    # Return average loss.
    mean_loss = tf.reduce_mean(tf.cast(loss, tf.float32))
    return mean_loss

# Define one training step.
@tf.function
def training_step(network, optimizer, images, labels):
    with tf.GradientTape() as tape:
        logits = network(images)
        loss = calculate_loss(logits, labels)

    gradients = tape.gradient(loss, network.trainable_variables)

    optimizer.apply_gradients(zip(gradients, network.trainable_variables))

    accuracy = calculate_accuracy(logits, labels)

    return loss, accuracy

# Define validation accuracy calculation.
# Why can't this be turned into a function? seems to cause issues with initializing mean_accuracy=None.
@tf.function
def validate_model(network, val_ds, steps_validation):
    mean_accuracy = None
    for val_images, val_labels in val_ds.take(steps_validation):
        logits = network(val_images)
        accuracy = calculate_accuracy(logits, val_labels)
        print(accuracy)
        if mean_accuracy is None:
            mean_accuracy = accuracy
        else:
            mean_accuracy += accuracy

    mean_accuracy /= steps_validation
    return mean_accuracy

# Define one training epoch.
def train_epoch(i_epoch, step_in_epoch, train_ds, val_ds, network, optimizer, BATCH_SIZE, checkpoint, network_folder, acc_history, loss_history):
    # Calculate the steps per epoch and validation loop based on the known dataset sizes. 
    
    # Why not get these from the *_ds inputs? Unfortunately, determining the actual size of these datasets is intentionally difficult, because the datasets are a deferred operation.
    l_train = tf.data.experimental.cardinality(train_ds)
    l_val = tf.data.experimental.cardinality(val_ds)

    steps_training = int(1281167 / BATCH_SIZE)
    steps_validation = int(50000 / BATCH_SIZE)
    i_step = 0

    # Step over the training dataset with steps_per_epoch.
    for train_images, train_labels in train_ds.take(steps_training):
        start = time.time()
        if step_in_epoch > steps_training: 
            break
        else: 
            step_in_epoch.assign_add(1)

        # Peform the training step for this batch.
        loss, acc = training_step(network, optimizer, train_images, train_labels)

        acc_history[i_epoch, i_step] = acc
        loss_history[i_epoch, i_step] = loss
        i_step += 1

        end = time.time()
        images_per_second = BATCH_SIZE / (end - start)
        
        # Save progress every so often.
        if step_in_epoch % 500 == 0:
            print(f'Checkpointed model at step {step_in_epoch.numpy()}.')
            checkpoint.save(network_folder + '/model')

        # Print progress every so often.
        if (step_in_epoch < 25) | (step_in_epoch % 100 == 0):
            print(f'Finished step {step_in_epoch.numpy()} of {steps_training} in epoch {i_epoch.numpy()}, loss={loss:.3f}, acc={acc:.3f} ({images_per_second:.3f} img/s).')

    # Save the network after every epoch:
    checkpoint.save(network_folder + '/model')

    # Compute the validation accuracy:
    
    # I would like to turn this into a function, but heaving mean_accuracy be a tensor throws things off.
    # mean_accuracy = validate_model(network, val_ds, steps_validation)
    mean_accuracy = None
    for val_images, val_labels in val_ds.take(steps_validation):
        logits = network(val_images)
        accuracy = calculate_accuracy(logits, val_labels)
        if mean_accuracy is None:
            mean_accuracy = accuracy
        else:
            mean_accuracy += accuracy

    mean_accuracy /= steps_validation

    print(f"Validation accuracy after epoch {i_epoch.numpy()}: {mean_accuracy:.4f}.")
    return acc_history, loss_history

def plot_figure(i_epoch,loss_history, acc_history, network_folder):
    dt_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    network_name = network_folder[0:-1]
    # Plot.
    plt.subplot(211)
    plt.plot(loss_history[i_epoch,:], color='blue')
    plt.title(f'Epoch {i_epoch.numpy()}: Loss')
    plt.xlabel("Iterations")
    plt.ylabel("Loss Value")
    # plot accuracy
    plt.subplot(212)
    plt.plot(100*acc_history[i_epoch,:], color='blue')
    plt.title(f'Epoch {i_epoch.numpy()}: Classification Accuracy')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy %")
    # save plot to file
    plt.savefig(f'{dt_string}_epoch_{i_epoch.numpy()}_{network_name}_plot.png')
    plt.close()
    print(f'Saved figure {dt_string}_epoch_{i_epoch.numpy()}_{network_name}_plot.png.')

# %%
def main():

    # Hyperparameters.
    BATCH_SIZE = 256
    # BATCH_SIZE = 1024 # Causes OOM.
    N_EPOCHS = 1
    LEARNING_RATE = 0.0001

    # Parse command line arguments.
    arg_help = "{0} -b <batch_size> -e <epochs> -l <learning_rate>".format(sys.argv[0])

    # Validate input.
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hb:e:l:", ["help", "batch_size=", "epochs=", "learning_rate="])
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
        elif opt in ("-b", "--batch_size"):
            BATCH_SIZE = int(arg)
        elif opt in ("-e", "--epochs"):
            N_EPOCHS = int(arg)
        elif opt in ("-l", "--learning_rate"):
            LEARNING_RATE = float(arg)

    network_folder = f'resnet34_bs_{BATCH_SIZE}_e_{N_EPOCHS}_lr_{LEARNING_RATE:.5f}/'

    print(f'Checkpointing model to {network_folder}.')

    # Get datasets.
    train_ds, val_ds = prepare_data_loader(BATCH_SIZE)

    # Show an example image.
    example_images, example_labels = next(iter(train_ds.take(1)))
    print("Initial Image size: ", example_images.shape)

    # Initialize network.
    network = ResNet34()
    output = network(example_images)
    print("Output shape:", output.shape)

    # Print network parameters.
    print("Network summary:")
    print(network.summary())

    # Define epoch and steps as state variables: this allows all TF operations on these variables.
    epoch = tf.Variable(
        initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
    step_in_epoch = tf.Variable(
        initial_value=tf.constant(0, dtype=tf.dtypes.int64),
        name='step_in_epoch')

    # Optimize with ADAM:
    print(f'Optimizing with ADAM, LR = {LEARNING_RATE:.5f}.')
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Define checkpoint format.
    checkpoint = tf.train.Checkpoint(
        network       = network,
        optimizer     = optimizer,
        epoch         = epoch,
        step_in_epoch = step_in_epoch)

    steps_training = int(1281167 / BATCH_SIZE)
    steps_validation = int(50000 / BATCH_SIZE)

    # Restore the model, if possible:
    latest_checkpoint = tf.train.latest_checkpoint(network_folder)
    if latest_checkpoint:
        print('Restoring checkpoint.')
        checkpoint.restore(latest_checkpoint)
        print('Checking accuracy on load.')
        # Compute the validation accuracy:
        mean_accuracy = None
        for val_images, val_labels in val_ds.take(steps_validation):
            logits = network(val_images)
            accuracy = calculate_accuracy(logits, val_labels)
            if mean_accuracy is None:
                mean_accuracy = accuracy
            else:
                mean_accuracy += accuracy

        mean_accuracy /= steps_validation

        print(f"Validation accuracy on load (epoch {epoch.numpy() - 1}, step {step_in_epoch.numpy()}): {mean_accuracy:.4f}.")
    else:
        print("Training fresh model.")

    acc_history = numpy.zeros((N_EPOCHS, steps_training))
    loss_history = numpy.zeros((N_EPOCHS, steps_training))

    # Training while loop. Datasets are automatically shuffled after each epoch.
    while epoch < N_EPOCHS:
        acc_history, loss_history = train_epoch(epoch, step_in_epoch, train_ds, val_ds, network, optimizer, BATCH_SIZE, checkpoint, network_folder, acc_history, loss_history)

        plot_figure(epoch, loss_history, acc_history, network_folder)

        epoch.assign_add(1)
        step_in_epoch.assign(0)

    # Final checkpoint after all epochs all done.
    checkpoint.save(network_folder + '/model')

# %%
# Actually run.
if __name__ == "__main__":
    main()

# %%
