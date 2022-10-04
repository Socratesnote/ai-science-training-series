
# %% Homework description
# In Class Exercises
# %% 1. In AI, datasets are often very large and cannot be processed all at once as is done in the loop above. The data is instead randomly sampled in smaller _batches_ where each _batch_ contains `batch_size` inputs. How can you change the loop above to sample the dataset in smaller batches? Hint: Our `data` variable is a Pandas `DataFrame` object, search for "how to sample a DataFrame".

# Answer: By changing the data 'collection' in the epoch loop to collect a subsample, e.g. data_sample = data.sample(batch_size), a smaller batch can be obtained and used for the update of the linear fit parameters.

# %% 2. As described above, learning rates that grow smaller over time can help find and get closer to global minima. In the loop above, our `learning_rate_m` and `learning_rate_b` are constant through the process of minimizing our parameters. How could you change the loop to reduce the learning rates over loop iterations?

# Answer: By changing the method of updating the learning rate. On each loop, scale the base learning rate by a normalization of the gradient to the maximal observed gradient thus far. This way, the initial learning rate is still taken into account, but if the current location on the gradient converges on a minimum, the step size is decreased.

# %% Homework
# Follow the example from the previous notebook [Linear Regression using SGD](./01_linear_regression_sgd.ipynb) and build a loop that properly finds the centers of these 4 clusters using k-means.
# Addendum: aim for an accuracy of > 90%.

# %% Linear regression
from random import randint, random, randrange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipydis
import time

# %% Load data.
data = pd.read_csv('slimmed_realestate_data.csv')
print(data.columns)

# %% Plot with Matplotlib.

figure = plt.figure(figsize=(6, 3))
ax = figure.add_axes([1, 1, 1, 1])
ax.set_title("Ground Living Area vs. Sale Price.")
scatter = ax.plot(data.GrLivArea, data.SalePrice, '.b')
ax.legend(scatter, ["Sale Price"])
ax.set_xlabel("Ground Living Area")
ax.set_ylabel("Sale Price")

# %% Manual regression.
n = len(data)
x = data['GrLivArea'].to_numpy()
y = data['SalePrice'].to_numpy()
# Define components of regression coefficient.
sum_xy = np.sum(x*y)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x*x)
denominator = n * sum_x2 - sum_x * sum_x
m = (n * sum_xy - sum_x * sum_y) / denominator
b = (sum_y * sum_x2 - sum_x * sum_xy) / denominator
print('Calculated linear fit: y = %f * x + %f' % (m, b))

# Saving these for later comparison.
m_calc = m
b_calc = b

# %% Define a linear estimator.


def get_lin_fit(x, m, b):
        # The 'arange' function generates points between two limits (min,max)
    linear_x = np.arange(x.min(), x.max())
    # Now use fit parameters to calculate the y points based on x points.
    linear_y = linear_x * m + b
    return linear_x, linear_y

# %% Define a plotting function. This will plot the data points and manually calculated linear fit.


def plot_data(x, y, m, b, ax):
    # Plot data points with 'bo' = blue circles.
    # Note that plot() returns tuples, so the comme disentangles that.
    scatter_plot, = ax.plot(x, y, 'bo')
    # Create the line based on linear fit parameters.
    linear_x, linear_y = get_lin_fit(x, m, b)
    # Plot the linear points using 'r-' = red line
    lin_plot, = ax.plot(linear_x, linear_y, 'r-')
    return scatter_plot, lin_plot

# %% Define model


def model(x, m, b):
    return m * x + b

# %% Define loss function


def loss(x, y, m, b):
    # Get predicted y-values from input and fit parameters.
    y_predicted = model(x, m, b)
    # Loss is the square of actual minus prediction, to obtain a smooth loss function.
    return np.power(y - y_predicted, 2)

# %% Define update dunctions


def updated_m(x, y, m, b, learning_rate, do_scale=False):
    # Gradients of loss function to parameters are manually calculated.
    dL_dm = np.mean(- 2 * x * (y - model(x, m, b)))
    new_m = m - learning_rate * dL_dm
    return new_m, dL_dm


def updated_b(x, y, m, b, learning_rate, do_scale=False):
    # Gradients of loss function to parameters are manually calculated.
    dL_db = np.mean(- 2 * (y - model(x, m, b)))
    new_b = b - learning_rate * dL_db
    return new_b, dL_db

# %% Define GD loop.
def gd_loop(loop_n, data, batch_size, m, b, learning_rate_m, learning_rate_b, do_plot, do_scale):
    # Track history of loss over time.
    loss_history = []

    # Initialize learning rates for scaled learning.
    learning_rate_m_base = learning_rate_m
    learning_rate_b_base = learning_rate_b

    # Initialize max gradients for scaled learning.
    dL_dm_max = 1
    dL_db_max = 1

    for i in range(loop_n):

        # Convert panda data to numpy arrays, one for the "Ground Living Area" and one for "Sale Price". Get a batch of B data points.
        data_sample = data.sample(batch_size)
        data_x = data_sample['GrLivArea'].to_numpy()
        data_y = data_sample['SalePrice'].to_numpy()

        # Update slope and intercept based on the current values.
        m, dL_dm = updated_m(data_x, data_y, m, b, learning_rate_m)
        b, dL_db = updated_b(data_x, data_y, m, b, learning_rate_b)

        if do_scale:
            # Update learning rates.
            dL_dm_max = max([dL_dm_max, abs(dL_dm)])
            dL_db_max = max([dL_db_max, abs(dL_db)])
            learning_scale_m = abs(dL_dm/dL_dm_max)
            learning_rate_m = learning_rate_m_base * learning_scale_m
            learning_scale_b = abs(dL_db/dL_db_max)
            learning_rate_b = learning_rate_b_base * learning_scale_b

        # Calculate the loss value for the new parameters.
        loss_value = np.mean(loss(data_x, data_y, m, b))

        # Store loss.
        loss_history.append(loss_value)

        if do_plot:
            # Print progress.
            print('[%03d] dy_i = %.2f * x + %.2f . Loss: %.1f'
            % (i, m, b, loss_value))
        
            # Create plot. Unfortunately, updating the plot isn't straightforward.
            # Create a 1 by 2 plot grid.
            fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=80)

            # Plot prediction and original.
            scatter_plot, lin_plot = plot_data(data_x, data_y, m, b, ax[0])
            ax[0].set_xlabel('Ground Living Area')
            ax[0].set_ylabel('Sale Price')

            # Here we also plot the calculated linear fit for comparison
            calc_x = np.arange(data_x.min(), data_x.max())
            calc_y = calc_x * m_calc + b_calc
            ax[0].plot(calc_x, calc_y, 'b-')
            ax[0].legend(["Data", "Learned Linear Fit", "Manual Linear Fit"])

            # Plot the loss.
            loss_x = np.arange(0, len(loss_history))
            loss_y = np.asarray(loss_history)
            ax[1].clear()
            ax[1].plot(loss_x, loss_y)
            ax[1].set_yscale('log')
            ax[1].set_xlabel('loop step')
            ax[1].set_ylabel('loss')
            ax[1].legend(["Loss"])
            plt.show()
            # Gives us time to see the plot
            time.sleep(0.01)
            # Clears the terminal output when the next plot is ready to show.
            ipydis.clear_output(wait=True)
    return loss_history, m, b


# %% Unscaled Learning.
# Initialize with random slope and intercept.
m = randrange(1, 25)
b = randrange(100, 10000, 100)

batch_size = 60
# Set a learning rate for each parameter individually.
learning_rate_m = 1e-7
learning_rate_b = 1e-1

# Run for n epochs.
loop_n = 30

# Set plotting and scaling toggles.
do_plot = False
do_scale = False

loss_history, m, b = gd_loop(loop_n, data, batch_size, m, b, learning_rate_m, learning_rate_b, do_plot, do_scale)


# %% Final
# Plot final learned fit vs manual fit.
# Create plot. Unfortunately, updating the plot isn't straightforward.
# Create a 1 by 2 plot grid.
fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=80)

# Plot prediction and original.
scatter_plot, lin_plot = plot_data(x, y, m, b, ax[0])
ax[0].set_xlabel('Ground Living Area')
ax[0].set_ylabel('Sale Price')

# Here we also plot the calculated linear fit for comparison
lin_x, lin_y = get_lin_fit(x, m_calc, b_calc)
ax[0].plot(lin_x, lin_y, 'b--')
ax[0].legend(["Data", "Final Unscaled Learned Linear Fit", "Manual Linear Fit"])

loss_x = np.arange(0, len(loss_history))
loss_y = np.asarray(loss_history)
ax[1].clear()
ax[1].plot(loss_x, loss_y)
ax[1].set_yscale('log')
ax[1].set_xlabel('loop step')
ax[1].set_ylabel('loss')
ax[1].legend(["Loss"])


# %% Scaled Learning.
# Initialize with random slope and intercept.
m = randrange(1, 25)
b = randrange(100, 10000, 100)

batch_size = 60
# Set initial learning rate for each parameter individually.
learning_rate_m = 1e-7
learning_rate_b = 1e-1

# Run for n epochs.
loop_n = 500

# Set plotting and scaling toggles.
do_plot = False
do_scale = True

loss_history, m, b = gd_loop(loop_n, data, batch_size, m, b, learning_rate_m, learning_rate_b, do_plot, do_scale)


# %% Final
# Plot final learned fit vs manual fit.
# Create plot. Unfortunately, updating the plot isn't straightforward.
# Create a 1 by 2 plot grid.
fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=80)

# Plot prediction and original.
scatter_plot, lin_plot = plot_data(x, y, m, b, ax[0])
ax[0].set_xlabel('Ground Living Area')
ax[0].set_ylabel('Sale Price')

# Here we also plot the calculated linear fit for comparison
lin_x, lin_y = get_lin_fit(x, m_calc, b_calc)
ax[0].plot(lin_x, lin_y, 'b--')
ax[0].legend(["Data", "Final Scaled Learned Linear Fit", "Manual Linear Fit"])

loss_x = np.arange(0, len(loss_history))
loss_y = np.asarray(loss_history)
ax[1].clear()
ax[1].plot(loss_x, loss_y)
ax[1].set_yscale('log')
ax[1].set_xlabel('loop step')
ax[1].set_ylabel('loss')
ax[1].legend(["Loss"])

# %% K-means clustering.


# %%