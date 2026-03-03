import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def load_dataset():
    dataset = pd.read_csv('IRIS_PLANT.csv')
    X = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = dataset[['species']]
    return X, y

def single_euclidean_distance(X_train: pd.DataFrame, instance: pd.DataFrame):
    distances = []
    X_train_np = X_train.to_numpy()
    instance_np = instance.to_numpy()

    for x in X_train_np:
        distances.append(np.sqrt(np.sum(np.square(instance_np - x))))
    return distances

X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_test)
for i in range(len(X_test)-1):
    print(single_euclidean_distance(X_train, X_test.iloc[i]))
# single_euclidean_distance(X_train, X_test.iloc[0])