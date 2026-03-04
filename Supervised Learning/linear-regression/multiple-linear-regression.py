import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


def create_multiple_lr_regression():
    # Declaring the given data
    data = {
        'hours': [5, 20, 5, 15, 10, 8, 12, 18, 6, 14, 9, 16, 7, 11, 13, 4,
                  17, 19, 3, 10, 6, 14, 8, 12, 15, 7, 9, 11, 13, 16],
        'homework': [8, 2, 9, 11, 10, 7, 10, 3, 8, 12, 9, 5, 6, 11, 8, 10, 4,
                   1, 12, 7, 9, 6, 8, 10, 5, 11, 7, 9, 12, 4],
        'grade': [10, 30, 20, 25, 15, 18, 22, 28, 12, 24, 17, 26, 14,
                         23, 21, 8, 27, 32, 6, 19, 13, 25, 16, 20, 29, 15, 18, 12, 24, 31]
    }
    # Create a pandas data frame with the dictionary
    df = pd.DataFrame(data)
    X = df[['hours', 'homework']]
    y = df['grade']

    # Splitting the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=42)
    # Training the model with the appropriate data
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Run the prediction over the testing data
    prediction = model.predict(X_test)
    print("\n----MODEL PARAMETERS----")
    print(f"θ₀ (Intersection): {model.intercept_:.5f}")
    print(f"θ₁ (Study weight): {model.coef_[0]:.5f}")
    print(f"θ₂ (Homework weight): {model.coef_[1]:.5f}")

    print("\n----RESULTANT EQUATION----")
    print(f"Equation: y = {model.intercept_:.5f} + {model.coef_[0]:.5f}(X₁) + {model.coef_[1]:.5f}(X₂)")

    print("\n----EVALUATION METRICS----")
    print(f"MSE: {mean_squared_error(y_test, prediction):.5f}")
    print(f"RMSE: {root_mean_squared_error(y_test, prediction):.5f}")
    print(f"R²: {r2_score(y_test, prediction):.5f}\n")

    # Calling the functions tha create the table and the charts
    create_table(df, model)
    create_charts(df, X_train, y_train, X_test, y_test, model)


def create_table(df: pd.DataFrame, model):
    # We calculate the prediction over all the data
    predictedGrade = model.predict(df[['hours', 'homework']])
    # Calculates the difference between the predicted grade and the actual grade
    diffActualPredicted = df['grade'] - predictedGrade
    # Creates two new columns
    df = df.assign(predictedGrade = predictedGrade)
    df = df.assign(diffActualPredicted = diffActualPredicted)
    # Modifies the column's names just to improve readability
    df.columns = ['Study hours', 'Homeworks completed', 'Real grade', 'Predicted grade', 'Actual - Predicted']
    print(df.to_string(index=False, justify='center'))


def set_axes_labels(ax: plt.Axes):
    # Recieves an Axes class and sets the labels
    ax.set_xlabel('Hours')
    ax.set_ylabel('Homework')
    ax.set_zlabel('Grade')


def create_charts(df, X_train, y_train, X_test, y_test, model):
    # 3D Chart showing the original data
    ax1 = plt.figure().add_subplot(projection='3d')

    ax1.scatter(df['hours'], df['homework'], df['grade'], marker='o')
    set_axes_labels(ax1)

    plt.title("Original data")
    plt.show()

    # 3D Chart showing the 80-20 proportion
    ax2 = plt.figure().add_subplot(projection='3d')

    ax2.scatter(X_train['hours'], X_train['homework'], y_train, marker='o', c='green', label='Train data')
    ax2.scatter(X_test['hours'], X_test['homework'], y_test, marker='*', c='orange', label='Test data')
    ax2.legend()
    set_axes_labels(ax2)

    plt.title("Train data vs Test data")
    plt.show()

    # 3D chart showing the plane over the original data
    ax3 = plt.figure().add_subplot(projection='3d')

    ax3.scatter(df['hours'], df['homework'], df['grade'], marker='o')

    # Making our data smoother for a better plane
    X_mesh = np.linspace(min(df['hours']), max(df['hours']), 20)
    Y_mesh = np.linspace(min(df['homework']), max(df['homework']), 20)
    
    # Creates a tuple of coordinate matrices
    X_mesh, Y_mesh = np.meshgrid(X_mesh, Y_mesh)
    
    # We calculate the dependent variable for each data in the 20*20 matrices
    Z_mesh = model.intercept_ + model.coef_[0] * X_mesh + model.coef_[1] * Y_mesh

    ax3.plot_surface(X_mesh, Y_mesh, Z_mesh, color='green', alpha=0.4)
    set_axes_labels(ax3)

    plt.show()


if __name__ == "__main__":
    create_multiple_lr_regression()