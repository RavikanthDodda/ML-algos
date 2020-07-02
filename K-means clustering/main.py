import scipy.io
from math import sqrt
import random
from copy import deepcopy
import matplotlib.pyplot as plt

# Loading matlab file.
Numpyfile = scipy.io.loadmat("AllSamples")

# Selecting data from dictionary.
dataset = Numpyfile['AllSamples']

# Function for calculating euclidean distance. Takes two 2D lists of length 1 as arguments.


def euclidean_dist(i, point):
    return sqrt((i[0]-point[0])**2 + (i[1]-point[1])**2)

# implementation of strategy 1 and strategy 2
# Function to calculate initial centroids prior to k means clustering.
# Depending on mode argument initialization strategy is selected
# Returns initial centroids


def initiate_centroids(mode, k, dataset):
    centroids = []
    #Strategy 1
    if mode == "random":
        while(k>0):
            val = True
            r = random.choice(dataset)
            for centroid in centroids:
                if r[0] == centroid[0] and r[1]== centroid[1]:
                    val = False
            if val:
                centroids.append(random.choice(dataset))
                k-=1
    #Strategy 2
    elif mode == "maximal distance":
        centroids.append(random.choice(dataset))
        while(k>1):
            new_centroid = []
            max_dist = 0
            for point in dataset:
                same = False
                avg = sum([euclidean_dist(c, point)
                           for c in centroids])/len(centroids)
                if avg > max_dist:
                    for c in centroids:
                        if point[0]==c[0] and point[1]==c[1]:
                            same = True
                    if not same:
                        max_dist = avg
                        new_centroid = point
            if len(new_centroid)!=0:
                k -= 1
                centroids.append(new_centroid)
    return centroids

# Updates centroids by moving them to mean of clusters.
# Performed at end of each step of k means clustering.
# Returns updated centroids


def update_centroids(centroids, clusters):
    for i in range(len(centroids)):
        if len(clusters[i]) > 0:
            centroids[i][0] = sum([j[0] for j in clusters[i]])/len(clusters[i])
            centroids[i][1] = sum([j[1] for j in clusters[i]])/len(clusters[i])
    return centroids

# Runs kmeans algortihm till convergence is reached.
# Takes initail centroids and dataset as arguments.
# Returns final centriods along with their corresponding clusters.


def kmeans(centroids, dataset):
    u_centroids = list()
    clusters = list()
    converged = False

    while(not converged):
        converged = True
        u_centroids, clusters = iteration(deepcopy(centroids), dataset)
        for i in range(len(u_centroids)):
            if centroids[i][0] != u_centroids[i][0] or centroids[i][1] != u_centroids[i][1]:
                converged = False
        centroids = u_centroids
    return u_centroids, clusters

# Function to perform a single iteration of steps in k means.
# Groups all points to their closest centroids and then updates centroids
# Returns updated centroids and the correspoding clusters


def iteration(centroids, dataset):
    clusters = []
    for i in centroids:
        clusters.append([])
    for point in dataset:
        min_dist = 99999
        cluster = 0
        for i in range(len(centroids)):
            d = euclidean_dist(centroids[i], point)
            if d < min_dist:
                min_dist = d
                cluster = i
        clusters[cluster].append(point)
    return update_centroids(centroids, clusters), clusters

# Function to calculate objective function value.
# Takes centroids and the corresponding clusters as arguments.
# Returns objective function value.


def objective_function(centroids, clusters):
    sum = 0
    for i in range(len(centroids)):
        for point in clusters[i]:

            sum += (centroids[i][0]-point[0])**2 + (centroids[i][1]-point[1])**2
    return sum


# Running Strategy 1
print("Strategy 1:")
for i in range(2):
    obf_val = list()
    centroids = list()
    clusters = list()
    print(" Run ", i+1, ":", "\n   k        objective function value")
    for k in range(2, 11):
        # Initiating centroids
        centroids = initiate_centroids('random', k, dataset)
        centroids, clusters = kmeans(centroids, dataset)
        # Calculating obj
        obf_val.append(objective_function(centroids, clusters))
        print("  ", k, "       ", obf_val[-1])
    # Plotting k vs objective function values.
    plt.plot([i for i in range(2, 11)], obf_val)
    plt.title(label="Strategy 1 Run "+str(i+1))
    plt.xlabel('k')
    plt.ylabel('objective function value')
    plt.show()

# Running Strategy 2
print("Strategy 2:")
for i in range(2):
    obf_val = list()
    centroids = list()
    clusters = list()
    print(" Run ", i+1, ":", "\n   k        objective function value")
    for k in range(2, 11):
        # Initiating centroids
        centroids = initiate_centroids('maximal distance', k, dataset)
        centroids, clusters = kmeans(centroids, dataset)
        # Calculating obj
        obf_val.append(objective_function(centroids, clusters))
        print("  ", k, "       ", obf_val[-1])
    # Plotting k vs objective function values.
    plt.plot([i for i in range(2, 11)], obf_val)
    plt.title(label="Strategy 2 Run "+str(i+1))
    plt.xlabel('k')
    plt.ylabel('objective function value')
    plt.show()
