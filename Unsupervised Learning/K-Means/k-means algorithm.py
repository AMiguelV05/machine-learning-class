import numpy as np
import pandas as pd


def load_data(path='IRIS_PLANT_NO_CLASSES.csv'):
    X = pd.read_csv(path)
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


def main():
    X = load_data()
    centroids = select_centroids(X, 3)
    distances = []
    for centroid in centroids:
        distances.append(single_euclidean_distance(X, centroid))
    distances = np.array(distances)
    print(X)
    print(np.argmin(distances, axis=0))



main()
