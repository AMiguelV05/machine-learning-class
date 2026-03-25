import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    X = pd.read_csv('IRIS_PLANT_NO_CLASSES.csv')
    return X


def select_centroids(X: pd.DataFrame, k):
    # Selects k unique samples
    selected_centroids = X.sample(n=k)

    return selected_centroids.to_numpy()

# Calculates the euclidian distance from a single centroid to all the points
def single_euclidean_distance(X: pd.DataFrame, centroid):
    distances = []
    XNumPy = X.to_numpy()

    for x in XNumPy:
        # For each instance in X, we calculate the distance to our centroid
        distances.append(np.sqrt(np.sum(np.square(centroid - x))))
    return distances


def update_centroids(X: pd.DataFrame, closerIndexes: np.ndarray, k):
    newCentroids = []
    XNumPy = X.to_numpy()

    for i in range(k):
        # Assigns all the true cases to the array
        pointsInCluster = XNumPy[closerIndexes == i]
        # If pointsInCluster has elements, it calculates the new centroid
        if len(pointsInCluster) > 0:
            newCentroids.append(np.mean(pointsInCluster, axis=0))
        # Otherwise, it takes a new sample
        else:
            newCentroids.append(X.sample(n=1).to_numpy()[0])
            print("New centroid created")

    return np.array(newCentroids)


def cluster_definition(X, centroids):
    k = len(centroids)
    centroidUpdatedTimes = 0
    closerIndexes = []

    while centroidUpdatedTimes < 100:
        distances = []
        # Calculates the distances for all centroids
        for centroid in centroids:
            distances.append(single_euclidean_distance(X, centroid))
        distances = np.array(distances)
        # Calculates the minimal distance centroids-points by columns
        closerIndexes = np.argmin(distances, axis=0)

        previous_centroids = centroids
        centroids = update_centroids(X, closerIndexes, k)
        centroidUpdatedTimes += 1

        # If the previous centroid is equal to the new one, it means that we have our best centroids
        if np.array_equal(previous_centroids, centroids):
            break

    return centroids, closerIndexes


def create_charts(X: pd.DataFrame, finalCentroids, closerIndexes):
    XNumPy = X.to_numpy()

    # Scatters the entire 0, 1 columns and colors it by the closerIndexes
    plt.scatter(XNumPy[:, 0], XNumPy[:, 1], c=closerIndexes, cmap='cool')
    # Scatters the entire 0, 1 columns of our centroids
    plt.scatter(finalCentroids[:, 0], finalCentroids[:, 1], color='black', label='Final centroids')
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title("Iris plant clusters")
    plt.legend()
    plt.show()


def main():
    X = load_data()
    k = 3
    centroids = select_centroids(X, k)
    finalCentroids, closerIndexes = cluster_definition(X, centroids)
    create_charts(X, finalCentroids, closerIndexes)


if __name__ == "__main__":
    main()
