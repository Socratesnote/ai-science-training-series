# %% [markdown]
# # Modern Neural Networks
# 
# In this tutorial, we'll cover "modern" neural networks.  In this case that specifically means deep, residual, convolutional neural networks.  Notably, the field of machine learning is "moving on" a bit these days from convoltutional neural networks.  The latest models are what are called "transformers" - we won't cover them today or likely at all in this course, but they certainly claim to be the next Big Thing.
# 
# ## Today's tutorial Agenda
# 
# We'll cover 3 things in the tutorial-focused portion of today's session:
# 
# 1) ImageNet Dataset, from a high view.  (More next week!)
# 
# 2) Recap of Convolutions, and the vanishing gradient problem
# 
# 3) Residual Layers and the ResNet development.
# 
# ## 1) ImageNet from a Mile High
# 
# In the 2010s, there was one dataset to rule them all: [ImageNet](https://www.image-net.org/).  We will use this dataset for the rest of this series since it's the territory of "Big Data."  The dataset is just about 200GB on disk, and contains 1.4M images to classify spread over 1000 classes.  Modern datasets from science are actually growing even bigger!  For today, we will use the data loading as a *black box*, paying no heed to how we're loading the data or what it's doing.  We will circle back next week, however, to get into this.

# %%
import tensorflow as tf
import json

# %%
# What's in this function?  Tune in next week ...
from ilsvrc_dataset import get_datasets

# We don't have to worry about the dataloader for now: that's next week's problem.

# %%
class FakeHvd:
    
    def size(self): return 1
    
    def rank(self): return 0


with open("ilsvrc.json", 'r') as f: 
    config = json.load(f)

print(json.dumps(config, indent=4))
    
    
config['hvd'] = FakeHvd()



# %%
train_ds, val_ds = get_datasets(config)

# %%
images, labels = next(iter(train_ds.take(1)))

# %%
print(images.shape)
print(labels.shape)

# %%
first_image = images[0]

# %%
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
# Divide by 255 to put this in the range of 0-1 for floating point colors
plt.imshow(first_image/255.) 

# %% [markdown]
# ## What is a convolution doing, again?
# 
# Convolution kernels are operating on entire images in small patches.  
# 
# ![Convolution Kernel](https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_17A-ConvolutionalNeuralNetworks-WHITEBG.png)
# 
# 
# That's a **single** convolution.  Convolutional Layers are learning multiple filters:
# 
# ![Convolution Layer](https://miro.medium.com/max/1400/1*u2el-HrqRPVk7x0xlvs_CA.png)

# %%
# Using 3 output filters here to simulate RGB.  
# You can - and SHOULD - use more for bigger networks
sample_conv_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=1)

# %%
# Apply this layer to all images:
modified_output = sample_conv_layer(images)

# %%
first_image = modified_output[0]
# Divide by 255 to put this in the range of 0-1 for floating point colors
plt.imshow(first_image/255.) 

# %% [markdown]
# This image is just as crisp as the original, but has had it's colors totally shifted.  That's expected: the convolution kernel size was just 1x1, or one pixel at a time.  So it's taking, for every pixel, the RGB value times a kernel (in this case, a vector):
# 
# $$ output = R*k_1 + G*k_2 + B*k_3$$
# 
# 
# More generally, this becomes a sum over neighboring pixels (for kernel sizes > 1).
# 
# We also produced 3 output "filters" - here, RGB again, but it can be more!  Each output filter for a convolution layer will create a $k x k$ kernel for every input filter, that are all summed together.  The total number of parameters is then:
# 
# $$ n_{params} = N_{Input Filters} \times N_{Output Filters} \times k_{x} \times k_{y} $$

# %%
sample_conv_layer_7x7 = tf.keras.layers.Conv2D(filters=3, kernel_size=7)
modified_output_7x7 = sample_conv_layer_7x7(images)

# %%
first_image = modified_output_7x7[0]
# Divide by 255 to put this in the range of 0-1 for floating point colors
plt.imshow(first_image/255.) 

# %% [markdown]
# This time the output is much blurrier - because this kernel has a 7x7 pixel size instead of a 1x1 pixel size.

# %% [markdown]
# ### Strides, Padding, Output Size
# 
# To apply a convolution, one algorithm takes the first output pixel to be the one where the filter just fits into the top left corner of the input image, and scans over (and then down) one pixel at a time.  There is nothing special about that though!  Kernels can skip pixels to reduce the output image size (sometimes called an downsampling convolution) and they can start with incomplete corners of the input images (padding with 0) to preserve the same output size.
# 
# - **Padding** represents the operation of handling the corner and edge cases so the output image is the same size as the input image.  Often you will see "valid" (apply the kernel only in valid locations) or "same" (add padding to make sure the output is the same size).
# 
# - **Stride** represents how many pixels are skipped between applications of the convolution.
# 
# - **Bottleneck** Layers are special convolution layers that have kernel size = 1, stride = 1 that preserve the output size and only look at single pixels at at time - though they look at all filters on a pixel.  A bottleneck layer is mathematically the same as applying the same MLP to each individual pixel's filters.

# %% [markdown]
# ## The case for ResNet: Vanishing Gradients
# 
# One of the motivations for the network we'll develop in the second half is the so-called "vanishing gradient problem":  The gradient of each layer depends on the gradient of each layer after it (remember the gradients flow backwards through the network).  Deeper and deeper networks that stack convolutions end up with smaller and smaller gradients in early layers.

# %%
class DenseLayer(tf.keras.Model):
    
    def __init__(self, n_filters):
        tf.keras.Model.__init__(self)
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )
        
    def __call__(self, inputs):
        
        x = inputs
        
        output1 = self.conv1(inputs)
        
        output1 = tf.keras.activations.sigmoid(output1)
        
        return output1
    
class ResidualLayer(tf.keras.Model):
    
    def __init__(self, n_filters):
        tf.keras.Model.__init__(self)
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )
    
    def __call__(self, inputs):
        
        x = inputs
        
        output1 = self.conv1(inputs)
        
        output1 = tf.keras.activations.sigmoid(output1)
        
        output2 = self.conv2(output1)

        return tf.keras.activations.sigmoid(output2 + x)
        

# %%
regular_layers  = [ DenseLayer(3) for i in range(100)]
residual_layers = [ ResidualLayer(3) for i in range(50)] # 2 convolutions per layer, so do half!

# %% [markdown]
# Apply these layers to the input data, and then compute a loss value (even it it's totally artificial!)

# %%
with tf.GradientTape() as tape:
    output = images
    for layer in regular_layers:
        output = layer(output)
    regular_loss = tf.reduce_mean(output)

# %% [markdown]
# Compute the gradients per layer:

# %%
regular_params = [l.trainable_weights for l in regular_layers]
gradients = tape.gradient(regular_loss, regular_params)

# %% [markdown]
# Lets do the same with the residual layers:

# %%
with tf.GradientTape() as tape:
    output = images
    for layer in residual_layers:
        output = layer(output)
    residual_loss = tf.reduce_mean(output)

# %%
residual_params = [l.trainable_weights for l in residual_layers]
residual_gradients = tape.gradient(residual_loss, residual_params)

# %%
# Compute the ratio of the gradient to the weights for each layer:
regular_mean_ratio = []
for layer, grad in zip(regular_params, gradients):
    regular_mean_ratio.append(tf.reduce_max(grad[0]) / tf.reduce_max(layer[0]))
    
plt.plot(range(len(regular_mean_ratio)), regular_mean_ratio)
plt.grid()
plt.yscale("log")

# %%
# Compute the ratio of the gradient to the weights for each layer:
residual_mean_ratio = []
for layer, grad in zip(residual_params, residual_gradients):

    residual_mean_ratio.append(tf.reduce_max(grad[0]) / tf.reduce_max(layer[0]))
    residual_mean_ratio.append(tf.reduce_max(grad[2]) / tf.reduce_max(layer[2]))

plt.plot(range(len(residual_mean_ratio)), residual_mean_ratio)
plt.yscale("log")
plt.grid()

# %% [markdown]
# The difference in the magnitude of the gradients is striking.  Yes, there are ways to keep the magnitude of the gradients more reasonable through normalization layers (and that helps in residual networks too!), but most cases that use residual connections show significant benefits compared to regular convolutional neural networks, especially as the networks get deeper.

# %%



