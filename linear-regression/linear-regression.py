from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from tabulate import tabulate


def calculate_linear_regression():
    # Creating the independent array with arange()0-15 in the necessary structure (reshape)
    X = np.arange(0, 16).reshape(-1, 1)
    # Dependent array
    y = [12, 14, 17, 19, 22, 25, 27, 30, 32, 35, 42, 44, 46, 48, 51, 55]

    # Using train_test_split with an 80(train_size)-20(test_size) proportion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Creating the linear model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # We use the prediction function and capture m, b, and calculate the score.
    prediction = model.predict(X)
    print(f"Slope: {model.coef_[0]:.5f}")
    print(f"Intersection: {model.intercept_:.5f}")
    print(f"Score: {model.score(X_test, y_test):.5f}")

    # Calling the functions to create the charts and the comparison table
    create_charts(X, y, X_train, y_train, X_test, y_test, prediction)
    create_comparison_table(X, y, prediction)


def create_charts(X, y, X_train, y_train, X_test, y_test, prediction):
    # Showing the original data
    plt.scatter(X, y)
    plt.xlabel("Years of experience")
    plt.ylabel("Salary")
    plt.title("Original data")
    plt.show()

    # A chart that shows the proportion of training and testing data
    plt.scatter(X_train, y_train, color='blue', label='Train data')
    plt.scatter(X_test, y_test, color='red', label='Test data')
    plt.xlabel("Years of experience")
    plt.ylabel("Salary")
    plt.title("Train-Test data proportion")
    plt.legend()
    plt.show()

    # Drawing the linear function over the scattered data
    plt.plot(X, prediction, color='red', label='Prediction')
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.xlabel("Years of experience")
    plt.ylabel("Salary")
    plt.title("Final prediction")
    plt.legend()
    plt.show()


def create_comparison_table(X, y, prediction):
    # Table array made of arrays
    table = []
    # For each element in X (year), y (actual salary), prediction (predicted salary). We append 'em to the table array
    for year, actualSalary, predictedSalary in zip(X, y, prediction):
        table.append([year, actualSalary, predictedSalary])

    # We call the tabulate function within the print and add the table parameter,
    # the headers of our table, and it's style
    print(tabulate(table, headers=["Years", "Real salary", "Predicted salary"], tablefmt="fancy_grid"))


def main():
    calculate_linear_regression()

if __name__ == "__main__":
    main()
