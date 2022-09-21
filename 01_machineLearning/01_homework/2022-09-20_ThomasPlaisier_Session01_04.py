# %% Imports.
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import time
import IPython.display as ipydis

# %% Define clusters.
# Create n points per cluster.
npoints = 400
# Create N clusters.
N = 4
# Use make_blobs from sklearn to create the cluster points. This returns a list of nx2 coordinates for the cluster points, nx1 labels, and Nx2 coordinates for centers.
x, true_labels, true_centers = make_blobs(n_samples=npoints, centers=N,
                       cluster_std=0.60, random_state=0,
                       return_centers=True)
# print("X: ", x.shape, "Labels:", cluster_labels.shape, "Centers:", cluster_centers.shape)

# %% Show the original cluster and their centers.
# plt.plot(x[:, 0], x[:, 1],'b.')
plt.scatter(x[:, 0], x[:, 1], c=true_labels, s=40, cmap='viridis')
plt.plot(true_centers[:,0],true_centers[:,1],'rx')
plt.legend(["Points", "Centers"])
plt.title('True labels.')


# %%


def initialize_centroids(x,N):
   """Initialize centroids as a random selection of 4 points from all available cluster points."""
   x_indices = np.random.choice(np.arange(0,x.shape[0]),N)
   centroids = x[x_indices]
   return centroids

def get_new_centroids(x, labels, N_clusters):
    """returns the new centroids assigned from the points closest to them"""
    # Get the average x1 and x2 (axis=0 so vertical) coordinate of all cluster points with a given label.
    centroids = [x[labels==this_cluster].mean(axis=0) for this_cluster in range(N_clusters)]
    # Cast centroids to an Nx2 array.
    return np.array(centroids)

def assign_labels(x, c):
   # Distance is sqrt( (x - x')**2 + (y - y')**2 )
   # Centroids are shape [N,2]
   # x is shape [npoints,2]
   # Add middle index to centroids to properly broadcast in math operations.
   c = c[:,np.newaxis,:] # [N,1,2]
   
   # Calculate (x - x')**2 and (y - y')**2
   # x is shape [npoints,2], c is [N,1,2], results in an array of shape: [N,npoints,2]
   dist2 = (x - c)**2
   
   # Calculate (x - x')**2 + (y - y')**2
   dist2 = dist2.sum(axis=2) # [N,npoints]
   
   # Out of the N distances, return the index (0-(N-1)) of the one that is the minimum distance away.
   label = np.argmin(dist2,axis=0) # [npoints]

   return label

def get_label_accuracy(true_labels, current_labels):
    accuracy = 100*np.sum((true_labels == current_labels).astype(int)) / true_labels.shape[0]
    return accuracy

def get_centroid_accuracy(true_centers, estimated_centers):

    # For each cluster in current_centroids, calculate the euclidian distance to each true center. Then pick the smallest distance. Accuracy is defined as the average of 1 - 1/dist for all N clusters.
    accuracy = []

    for this_estimated in range(estimated_centers.shape[0]):
        this_center = estimated_centers[this_estimated,:]
        distances = np.sqrt(np.sum([np.power(this_center - true_centers[this_true, :], 2) for this_true in range(true_centers.shape[0])],axis=1))
        # Find minimal distance.
        min_index = np.argmin(distances)
        accuracy.append(1/(1 + distances[min_index]))
    
    accuracy = 100*np.mean(accuracy)
    return accuracy

# %% Initialize random centroids and labels.
c = initialize_centroids(x,N)
l = assign_labels(x,c)
# plt.scatter(x[:, 0], x[:, 1], c=l, s=40, cmap='viridis')
# plt.plot(cluster_centers[:,0],cluster_centers[:,1],'rx')
# plt.title("Initial state.")
delta = 0.00001
last_centroids = initialize_centroids(x,N)
last_labels = assign_labels(x,last_centroids)
epochs = 30
do_break = False

# %% Make a loop that performs proper k-means clustering.

for step in range(epochs):
    # Get newest centroid positions, and re-calculate labels.
    estimated_centroids = get_new_centroids(x,last_labels,N)
    estimated_labels = assign_labels(x,estimated_centroids)

    # Test if centroids have stopped moving, or no new labels are assigned. If either is stationary, update the other and break after plotting.
    if np.all((last_centroids - estimated_centroids) < delta):
        print('Centroids unchanged as of step %d.' % step)
        estimated_labels = assign_labels(x,estimated_centroids)
        do_break = True
    # elif np.all(labels == last_labels):
    # Don't break if labels do not change.
    #     print('Point labels unchanged as of step %d.' % step)
    #     centroids = get_new_centroids(x,labels,N)
    #     do_break = True
   
    last_labels = estimated_labels
    last_centroids = estimated_centroids

    # We can use the "truth" labels, cluster_labels, to see how well we are doing in terms of accuracy: # of points correctly labeled / total number of points. However, this entirely depends on which label was randomly assigned to a given centroid at the beginning.
    # accuracy = get_label_accuracy(true_labels, estimated_labels)

    # Perhaps a better way of defining accuracy is to determine the average of how far the N centroids are from the original values. To do this, find the closest true centroid for each of the estimated centroids, and take the average of all N centroids: as average distance goes to 0, accuracy goes to 100%.
    accuracy = get_centroid_accuracy(true_centers, estimated_centroids)

    plt.scatter(x[:, 0], x[:, 1], c=last_labels, s=40, cmap='viridis')
    plt.plot(estimated_centroids[:,0],estimated_centroids[:,1],'rx')
    plt.title('Step %d, Centroid Accuracy %.2f' % (step, accuracy))
    plt.show()
    # If still updating, sleep and clear.
    if do_break:
        break

    time.sleep(0.1)
    # Clears the terminal output when the next plot is ready to show.
    ipydis.clear_output(wait=True)

# %%