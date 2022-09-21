
# %% Homework description
# In Class Exercises
# 1. In AI, datasets are often very large and cannot be processed all at once as is done in the loop above. The data is instead randomly sampled in smaller _batches_ where each _batch_ contains `batch_size` inputs. How can you change the loop above to sample the dataset in smaller batches? Hint: Our `data` variable is a Pandas `DataFrame` object, search for "how to sample a DataFrame".
# 2. As described above, learning rates that grow smaller over time can help find and get closer to global minima. In the loop above, our `learning_rate_m` and `learning_rate_b` are constant through the process of minimizing our parameters. How could you change the loop to reduce the learning rates over loop iterations?
# Homework
# Follow the example from the previous notebook [Linear Regression using SGD](./01_linear_regression_sgd.ipynb) and build a loop that properly finds the centers of these 4 clusters using k-means.
# Addendum: aim for an accuracy of > 90%.

# %% Linear regression
from random import random, randrange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipydis
import time

# %% Load data.
data = pd.read_csv('slimmed_realestate_data.csv')
print(data.columns)

# %% Plot.

# Matplotlib
figure = plt.figure(figsize=(6, 3))
ax = figure.add_axes([1, 1, 1, 1])
ax.set_title("Ground Living Area vs. Sale Price.")
scatter = ax.plot(data.GrLivArea, data.SalePrice, '.b')
ax.legend(scatter, ["Sale Price"])
ax.set_xlabel("Ground Living Area")
ax.set_ylabel("Sale Price")

# %% Define regression
n = len(data)
x = data['GrLivArea'].to_numpy()
y = data['SalePrice'].to_numpy()
sum_xy = np.sum(x*y)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x*x)
denominator = n * sum_x2 - sum_x * sum_x
m = (n * sum_xy - sum_x * sum_y) / denominator
b = (sum_y * sum_x2 - sum_x * sum_xy) / denominator
print('y = %f * x + %f' % (m, b))

# Saving these for later comparison.
m_calc = m
b_calc = b

# %% Define a plotting function.


def plot_data(x, y, m, b, ax):
    # plot our data points with 'bo' = blue circles
    ax.plot(x, y, 'bo')
    # create the line based on our linear fit
    # first we need to make x points
    # the 'arange' function generates points between two limits (min,max)
    linear_x = np.arange(x.min(), x.max())
    # now we use our fit parameters to calculate the y points based on our x points
    linear_y = linear_x * m + b
    # plot the linear points using 'r-' = red line
    ax.plot(linear_x, linear_y, 'r-', label='fit')
    # Labels.
    ax.legend(["Data", "Linear prediction"])
    ax.set_xlabel("Ground Living Area")
    ax.set_ylabel("Sale Price")


# %%
plot_data(x, y, m, b, ax)

# %% Modelling

# %% Define model


def model(x, m, b):
    return m * x + b

# %% Define loss function


def loss(x, y, m, b):
    y_predicted = model(x, m, b)
    return np.power(y - y_predicted, 2)

# %% Update dunctions


def updated_m(x, y, m, b, learning_rate):
    dL_dm = - 2 * x * (y - model(x, m, b))
    dL_dm = np.mean(dL_dm)
    return m - learning_rate * dL_dm


def updated_b(x, y, m, b, learning_rate):
    dL_db = - 2 * (y - model(x, m, b))
    dL_db = np.mean(dL_db)
    return b - learning_rate * dL_db


# %% Initialize model
m = randrange(1, 25)
b = randrange(100, 10000, 100)
print('y_i = %.2f * x + %.2f' % (m, b))

# %% Get first losses.
# Note that this calculates the loss function over all data points for initial m,b.
l = loss(x, y, m, b)
print('first 10 loss values: ', l[:10])

# %% Try one loop.
learning_rate = 1e-9
m = updated_m(x, y, m, b, learning_rate)
b = updated_b(x, y, m, b, learning_rate)
print('y_i = %.2f * x + %.2f     previously calculated: y_i = %.2f * x + %.2f' %
      (m, b, m_calc, b_calc))
plot_data(x, y, m, b, ax)

# %% Learning loop.
# set our initial slope and intercept
m = randrange(1, 25)
b = randrange(100, 10000, 100)

# batch_size = 60
# set a learning rate for each parameter
learning_rate_m = 1e-7
learning_rate_b = 1e-1
# use these to plot our progress over time
loss_history = []

# we run our loop N times.
loop_N = 30
for i in range(loop_N):

    # convert panda data to numpy arrays, one for the "Ground Living Area" and one for "Sale Price"
    data_x = data['GrLivArea'].to_numpy()
    data_y = data['SalePrice'].to_numpy()

    # update our slope and intercept based on the current values
    m = updated_m(data_x, data_y, m, b, learning_rate_m)
    b = updated_b(data_x, data_y, m, b, learning_rate_b)

    # calculate the loss value
    loss_value = np.mean(loss(data_x, data_y, m, b))

    # keep a history of our loss values
    loss_history.append(loss_value)

    # print our progress
    print('[%03d]  dy_i = %.2f * x + %.2f     previously calculated: y_i = %.2f * x + %.2f    loss: %f' %
          (i, m, b, m_calc, b_calc, loss_value))

    # close/delete previous plots
    plt.close('all')

    # create a 1 by 2 plot grid
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=300)
    # plot our usual output
    plot_data(data_x, data_y, m, b, ax[0])

    # here we also plot the calculated linear fit for comparison
    line_x = np.arange(data_x.min(), data_x.max())
    line_y = line_x * m_calc + b_calc
    ax[0].plot(line_x, line_y, 'b-', label='calculated')
    # add a legend to the plot and x/y labels
    ax[0].legend()
    ax[0].set_xlabel('square footage')
    ax[0].set_ylabel('sale price')

    # plot the loss
    loss_x = np.arange(0, len(loss_history))
    loss_y = np.asarray(loss_history)
    ax[1].plot(loss_x, loss_y)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('loop step')
    ax[1].set_ylabel('loss')
    plt.show()
    # gives us time to see the plot
    time.sleep(0.5)
    # clears the plot when the next plot is ready to show.
    ipydis.clear_output(wait=True)

# %%
