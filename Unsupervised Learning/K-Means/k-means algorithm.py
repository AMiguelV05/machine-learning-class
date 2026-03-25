import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data():
    X = pd.read_csv(Path(__file__).parent / 'IRIS_PLANT_NO_CLASSES.csv')
    return X


def select_centroids(X: pd.DataFrame, k):
    selected_centroids = X.sample(n=k)
    return selected_centroids.to_numpy()


def single_euclidean_distance(X: pd.DataFrame, centroid):
    distances = []
    XNumPy = X.to_numpy()

    for x in XNumPy:
        # For each instance in X, we calculate the distance to our centroid
        distances.append(np.sqrt(np.sum(np.square(centroid - x))))
    return distances


def update_centroids(X: pd.DataFrame, clusters: np.ndarray, k):
    newCentroids = []
    XNumPy = X.to_numpy()
    print("\nSe actualizaron los centroides\n")
    for i in range(k):
        pointsInCluster = XNumPy[clusters == i]

        if len(pointsInCluster) > 0:
            newCentroids.append(np.mean(pointsInCluster, axis=0))
        else:
            newCentroids.append(X.sample(n=1).to_numpy()[0])
            print("Se tuvo que crear uno nuevo")

    return np.array(newCentroids)


def cluster_definition(X, centroids):
    centroidUpdatedTimes = 0
    while True or centroidUpdatedTimes < 100:
        distances = []
        for centroid in centroids:
            distances.append(single_euclidean_distance(X, centroid))
        distances = np.array(distances)
        winningIndexes = np.argmin(distances, axis=0)
        previous_centroids = centroids
        centroids = update_centroids(X, winningIndexes, 3)
        centroidUpdatedTimes += 1
        print(centroidUpdatedTimes)
        if np.array_equal(previous_centroids, centroids):
            break
    return centroids


def create_charts(X: pd.DataFrame, finalCentroids):
    plt.scatter(X['sepal_length'], X['sepal_width'], color='#00DDFF', label='Entire dataset')
    plt.scatter(finalCentroids[:, 0], finalCentroids[:, 1], color='#F700FF', label='Final centroids')
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title("Iris plant clusters")
    plt.legend()
    plt.show()


def main():
    X = load_data()
    centroids = select_centroids(X, 3)
    finalCentroids = cluster_definition(X, centroids)

    create_charts(X, finalCentroids)


if __name__ == "__main__":
    main()
