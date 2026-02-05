import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


def mainfunc():
    data = {
        'horas': [5, 20, 5, 15, 10, 8, 12, 18, 6, 14, 9, 16, 7, 11, 13, 4,
                  17, 19, 3, 10, 6, 14, 8, 12, 15, 7, 9, 11, 13, 16],
        'tareas': [8, 2, 9, 11, 10, 7, 10, 3, 8, 12, 9, 5, 6, 11, 8, 10, 4,
                   1, 12, 7, 9, 6, 8, 10, 5, 11, 7, 9, 12, 4],
        'calificacion': [10, 30, 20, 25, 15, 18, 22, 28, 12, 24, 17, 26, 14,
                         23, 21, 8, 27, 32, 6, 19, 13, 25, 16, 20, 29, 15, 18, 12, 24, 31]
    }
    df = pd.DataFrame(data)
    X = df[['horas', 'tareas']]
    y = df['calificacion']

    trainX, testX, trainY, testY = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(trainX, trainY)

    prediction = model.predict(testX)
    print("\n----MODEL PARAMETERS----")
    print(f"θ₀ (Intersection): {model.intercept_:.5f}")
    print(f"θ₁ (Study weight): {model.coef_[0]:.5f}")
    print(f"θ₂ (Homework weight): {model.coef_[1]:.5f}")
    print("\n----RESULTANT EQUATION----")
    print(f"Equation: y = {model.intercept_:.5f} + {model.coef_[0]:.5f}(X₁) + {model.coef_[1]:.5f}(X₂)")
    print("\n----EVALUATION METRICS----")
    print(f"MSE: {mean_squared_error(testY, prediction):.5f}")
    print(f"RMSE: {root_mean_squared_error(testY, prediction):.5f}")
    print(f"R²: {r2_score(testY, prediction):.5f}")
    create_charts(df)

def create_charts(df):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = df['horas']
    ys = df['tareas']
    zs = df['calificacion']

    ax.scatter(xs, ys, zs=zs, marker='o')
    ax.set_xlabel('horas')
    ax.set_ylabel('tareas')
    ax.set_zlabel('calificacion')
    plt.show()

mainfunc()