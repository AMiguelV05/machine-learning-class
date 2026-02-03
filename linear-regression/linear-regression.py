import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def calculate_linear_regression():
    # Creating the independent array with arange()0-15 in the necessary structure (reshape)
    X = np.arange(0, 16).reshape(-1, 1)
    # Dependent array
    y = [12, 14, 17, 19, 22, 25, 27, 30, 32, 35, 42, 44, 46, 48, 51, 55]

    # Showing the original data
    plt.scatter(X, y)
    plt.title("Original data")
    plt.show()

    # Using train_test_split with an 80(train_size)-20(test_size) proportion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # A graphic that shows the proportion of training and testing data
    plt.scatter(X_train, y_train, color='blue', label='Train data')
    plt.scatter(X_test, y_test, color='red', label='Test data')
    plt.title("Train-Test data proportion")
    plt.legend()
    plt.show()

    # Creating the linear model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # We use the prediction function and capture m, b.
    prediction = model.predict(X)
    print(f"Steep: {model.coef_}")
    print(f"Intersection: {model.intercept_}")

    # Drawing the linear function over the scattered data
    plt.plot(X, prediction, color='red', label='Prediction')
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.title("Final prediction")
    plt.legend()
    plt.show()

def main():
    calculate_linear_regression()

if __name__ == "__main__":
    main()
