import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter


# Function to load the CSV
def load_dataset():
    dataset = pd.read_csv('IRIS_PLANT.csv')
    # Separate the data into two different DataFrames
    # Data to process
    X = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    # Target class
    y = dataset[['species']]
    return X, y


# Calculates the euclidian distance from just one instance to all our training data
def single_euclidean_distance(x_train: pd.DataFrame, instance: pd.DataFrame):
    distances = []
    # Converts DataFrames to a NumPy array
    X_train_np = x_train.to_numpy()
    instance_np = instance.to_numpy()

    for x in X_train_np:
        # For each instance in X_train_np, we calculate the distance to our target instance
        distances.append(np.sqrt(np.sum(np.square(instance_np - x))))
    return distances


def find_k_neighbors_class(distances: list[np.float64], y_train: pd.DataFrame, k: int):
    # Transforms the list into a NumPy array in order to use its methods
    distances_np = np.array(distances)
    # Sorts the distances and returns a list of indexes from zero to k
    indexes = np.argsort(distances_np)[:k].tolist()
    classes = []
    for index in indexes:
        # For each index in indexes we obtain the corresponding value of the target class from our training DataFrame
        classes.append(y_train.iloc[index].values[0])
    # Uses the Counter class to easily detect the repeated values
    count = Counter(classes)
    # We use a method to find the most common element in count and return the pure value
    return count.most_common()[0][0]


def format_table(df: pd.DataFrame):
    # Formatting column names and printing the DataFrame without index
    df.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Real class', 'Predicted class']
    print(df.to_string(index=False, justify='center'))


def calculate_accuracy(df: pd.DataFrame):
    # This comparison returns a Pandas Series full of True and False
    coincidences = df['Real class'] == df['Predicted class']
    # Uses a method from the series class to calculate the accuracy,
    # the "mean" method gives us true values / total of values
    print(f"Accuracy: {coincidences.mean():.3f}")


def test_model():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=450)
    # Copies X_test values and y_test values to a new DataFrame
    resultantDataFrame = X_test.copy()
    resultantDataFrame['Real class'] = y_test.copy()

    predictedClasses = []
    # Asks for user input to assign k
    try:
        kTimes = int(input("Ingresa el valor de k: "))
        if kTimes <= 0:
            raise ValueError
    except ValueError:
        print("\nSe ha introducido un número inválido, se asignara el valor de 3 por defecto.\n")
        kTimes = 3

    for i in range(len(X_test)):
        # Captures the current instance by its id
        actualInstance = X_test.iloc[i]
        # Calls the single_euclidian_distance function with X_train and actualInstance as parameters
        distances = single_euclidean_distance(X_train, actualInstance)
        # Calls the find_k_neighbors_class function with distances, y_train and kTimes as parameters
        predictedClasses.append(find_k_neighbors_class(distances=distances, y_train=y_train, k=kTimes))
    # Assigns the predicted classes to a new column in the DataFrame
    resultantDataFrame['Predicted class'] = predictedClasses
    format_table(resultantDataFrame)
    calculate_accuracy(resultantDataFrame)


if __name__ == '__main__':
    test_model()