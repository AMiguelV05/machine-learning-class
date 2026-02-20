from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import pandas as pd
from matplotlib import pyplot as plt


def training_model():
    # Reading the data from the archive
    df = pd.read_csv('titanic_datos.csv')
    df = df.drop('PassengerId', axis=1)
    # We processed the data
    X, y = process_data(df)
    # Train split with a 70(train)-30(test) ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    # We create and train our model with overfitting
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    # Create a prediction for the overfitting tree over our X_test
    y_pred = dt.predict(X_test)

    # Calculate the metrics and print 'em in console
    print("\nMetrics for tree with overfitting:\n")
    calculate_metrics(y_test, y_pred, complementTitle='with overfitting', dt=dt, X=X, y_all_true=df['Survived'])
    create_tree(dt, complementTitle='without pruning', X=X)

    # We create and train our model with pre-pruning
    dt2 = DecisionTreeClassifier(max_depth=4, min_samples_split=3)
    dt2.fit(X_train, y_train)

    # Create a prediction for the pre-pruning tree over our X_test
    y_pred2 = dt2.predict(X_test)

    # Calculate the metrics and print 'em in console
    print("Metrics for tree with pre-pruning:\n")
    calculate_metrics(y_test, y_pred2, complementTitle='with pre-pruning', dt=dt2, X=X, y_all_true=df['Survived'])
    create_tree(dt2, complementTitle='with pre-pruning', X=X)
    create_table(df, dt, dt2, X)


def process_data(df):
    # Setting number values for attribute sex: male(1) and female(0)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    # All the attributes
    X = df.drop('Survived', axis=1)
    # Target class
    y = df['Survived']

    return X, y


def calculate_metrics(y_test, y_pred, complementTitle: str, dt, X, y_all_true):
    # Calculating the different metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("---Metrics---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    # We calculate the confusion matrix
    y_all_pred = dt.predict(X)
    confusionMatrix = confusion_matrix(y_all_true, y_all_pred)
    # Obtaining the values from the matrix
    tn, fp, fn, tp = confusionMatrix.ravel()
    # Calculating specificity
    specificity = tn / (tn + fp)
    print(f"Specificity: {specificity:.4f}\n")
    # Using the function to display the confusion matrix
    ConfusionMatrixDisplay(confusionMatrix, display_labels=['Not survived', 'Survived']).plot()
    plt.title(f'Confusion Matrix {complementTitle}')
    plt.show()


def create_table(df: pd.DataFrame, dt, dt2, X):
    # We recalculate the predictions with the entire dataset
    y_pred1 = dt.predict(X)
    y_pred2 = dt2.predict(X)
    # Creating two new columns
    df = df.assign(predicted1=y_pred1)
    df = df.assign(predicted2=y_pred2)
    # Re-naming columns
    df.columns = ['Class', 'Sex', 'Age', 'SibSp', 'Fare', 'Real survived',
                  'Predicted overfitting', 'Predicted pre-pruning']
    print(df.to_string(index=False, justify='center'))


def create_tree(dt, complementTitle: str, X):
    # We use the "plot_tree" function to display it in a chart
    plot_tree(
        dt,
        feature_names=X.columns,
        class_names=['Not survived', 'Survived'],
        filled=True)
    plt.title(f'Decision tree classifier {complementTitle}')
    plt.show()

if __name__ == '__main__':
    training_model()