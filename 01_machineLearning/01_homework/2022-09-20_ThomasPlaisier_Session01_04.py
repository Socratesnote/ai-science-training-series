# %% Imports.
from sklearn.datasets import make_blobs
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import IPython.display as ipydis

# %%


def initialize_centroids(x, N):
    """Initialize centroids as a random selection of 4 points from all available cluster points."""
    x_indices = np.random.choice(np.arange(0, x.shape[0]), N)
    centroids = x[x_indices]
    return centroids


def get_new_centroids(x, labels, N_clusters):
    """returns the new centroids assigned from the points closest to them"""
    # Get the average x1 and x2 (axis=0 so vertical) coordinate of all cluster points with a given label.
    centroids = [x[labels == this_cluster].mean(
        axis=0) for this_cluster in range(N_clusters)]
    # Cast centroids to an Nx2 array.
    return np.array(centroids)


def assign_labels(x, c):
    # Distance is sqrt( (x - x')**2 + (y - y')**2 )
    # Centroids are shape [N,2]
    # x is shape [npoints,2]
    # Add middle index to centroids to properly broadcast in math operations.
    c = c[:, np.newaxis, :]  # [N,1,2]

    # Calculate (x - x')**2 and (y - y')**2
    # x is shape [npoints,2], c is [N,1,2], results in an array of shape: [N,npoints,2]
    dist2 = (x - c)**2

    # Calculate (x - x')**2 + (y - y')**2
    dist2 = dist2.sum(axis=2)  # [N,npoints]

    # Out of the N distances, return the index (0-(N-1)) of the one that is the minimum distance away.
    label = np.argmin(dist2, axis=0)  # [npoints]

    return label

# OG Accuracy


def get_label_accuracy(true_labels, current_labels):
    accuracy = 100 * \
        np.sum((true_labels == current_labels).astype(int)) / \
        true_labels.shape[0]
    return accuracy

# Modified accuracy: shuffle current labels until the maximum overlap is reached.


def get_label_accuracy_mod(true_labels, current_labels):
    # Modified label accuracy: take the current label estimation, and swap labels until maximum overlap between true and current is reached: that is considered the accuracy.
    # Get list of all labels.
    all_labels = np.arange(0, N)
    init_acc = get_label_accuracy(true_labels, current_labels)
    new_labels = current_labels

    for this_from_label in all_labels:
        this_acc = []
        for this_to_label in all_labels:
            # Initialize.
            temp_labels = new_labels
            # Temp swap.
            from_idx = np.where(temp_labels == this_from_label)[0]
            to_idx = np.where(temp_labels == this_to_label)[0]
            # Swap labels
            temp_labels[from_idx] = this_to_label
            temp_labels[to_idx] = this_from_label
            this_acc.append(get_label_accuracy(true_labels, temp_labels))
        max_idx = this_acc.index(max(this_acc))
        # Final swap.
        from_idx = np.where(new_labels == this_from_label)[0]
        to_idx = np.where(new_labels == all_labels[max_idx])[0]
        # Swap labels
        new_labels[from_idx] = all_labels[max_idx]
        new_labels[to_idx] = this_from_label

    accuracy = get_label_accuracy(true_labels, new_labels)
    return accuracy, new_labels


def get_centroid_distance(true_centers, estimated_centers):

    # For each cluster in current_centroids, calculate the euclidian distance to each true center. Then pick the smallest distance.
    all_distances = []

    for this_estimated in range(estimated_centers.shape[0]):
        this_center = estimated_centers[this_estimated, :]
        distances = np.sqrt(np.sum([np.power(this_center - true_centers[this_true, :], 2)
                            for this_true in range(true_centers.shape[0])], axis=1))
        all_distances.append(min(distances))

    return all_distances



# %% Define clusters.
# Create n points per cluster.
npoints = 500
# Create N clusters.
N = 4
# Use make_blobs from sklearn to create the cluster points. This returns a list of nx2 coordinates for the cluster points, nx1 labels, and Nx2 coordinates for centers.
x, true_labels, true_centers = make_blobs(n_samples=npoints, centers=N,
                                          cluster_std=0.60, random_state=0,
                                          return_centers=True)

# %% Show the original cluster and their centers.
plt.scatter(x[:, 0], x[:, 1], c=true_labels, s=40, cmap='viridis')
plt.plot(true_centers[:, 0], true_centers[:, 1], 'rx')
plt.legend(["Points", "Centers"])
plt.title('True labels.')
if hasattr(sys, 'ps1'):  # True if interactive.
    plt.show()
else:
    # Run from terminal.
    plt.savefig('img/Truth.png', format='png')
    plt.cla()  # Clear axes to avoid ghosting.

# %% Initialize random centroids and labels.
last_centroids = initialize_centroids(x, N)
last_labels = assign_labels(x, last_centroids)
plt.scatter(x[:, 0], x[:, 1], c=last_labels, s=40, cmap='viridis')
plt.plot(last_centroids[:,0], last_centroids[:,1],'rx')
plt.title("Initial state.")
if hasattr(sys, 'ps1'):  # True if interactive.
    plt.show()
else:
    # Run from terminal.
    plt.savefig('img/Step_.png', format='png')
    plt.cla()  # Clear axes to avoid ghosting.

# %% Loop parameters
delta = 0.001
epochs = 20
do_break = False

# %% Make a loop that performs proper k-means clustering.

for step in range(epochs):
    # Get newest centroid positions, and re-calculate labels.
    estimated_centroids = get_new_centroids(x, last_labels, N)  
    estimated_labels = assign_labels(x, estimated_centroids)

    # Test if centroids have stopped moving, or no new labels are assigned. If either is stationary, update the other and break after plotting.
    if np.all((last_centroids - estimated_centroids) < delta):
        print('Centroids unchanged as of step %d.' % step)
        estimated_labels = assign_labels(x, estimated_centroids)
        do_break = True
    # elif np.all(labels == last_labels):
    # Don't break if labels do not change.
    #     print('Point labels unchanged as of step %d.' % step)
    #     centroids = get_new_centroids(x,labels,N)
    #     do_break = True

    last_labels = estimated_labels
    last_centroids = estimated_centroids

    # We can use the "truth" labels, cluster_labels, to see how well we are doing in terms of accuracy: # of points correctly labeled / total number of points. However, this entirely depends on which label was randomly assigned to a given centroid at the beginning.
    label_accuracy, dummy = get_label_accuracy_mod(
        true_labels, estimated_labels)

    # Perhaps a better way of defining accuracy is to determine the average of how far the N centroids are from the original values. To do this, find the closest true centroid for each of the estimated centroids, and take the average of all N centroids: as average distance goes to 0, accuracy goes to 100%.
    all_distances = get_centroid_distance(true_centers, estimated_centroids)

    plt.scatter(x[:, 0], x[:, 1], c=last_labels, s=40, cmap='viridis')
    plt.plot(estimated_centroids[:, 0], estimated_centroids[:, 1], 'rx')
    plt.title('Step %d, Label Acc %.2f, Centroid Dist %.2f' %
              (step, label_accuracy, np.mean(all_distances)))
    if hasattr(sys, 'ps1'):  # True if interactive.
        plt.show()
    else:
        # Run from terminal.
        plt.savefig('img/Step_%d.png' % (step), format='png')
        plt.cla()  # Clear axes to avoid ghosting.

    # If still updating, sleep and clear.
    if do_break:
        break

    time.sleep(0.5)
    # Clears the terminal output when the next plot is ready to show.
    if hasattr(sys, 'ps1'):  # True if interactive.
        ipydis.clear_output(wait=True)

# plt.show()
# %%
