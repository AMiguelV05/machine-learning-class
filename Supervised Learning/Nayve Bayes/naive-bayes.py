from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import pandas as pd
from matplotlib import pyplot as plt

# Function in charge to read the data from the xlsx file
def load_data():
    df = pd.read_excel("spam_dataset.xlsx")
    # Separates the main dataframe into two different in order to separate the target class
    X = df['Mensaje']
    y = df['Etiqueta']
    return X, y

def model_training(X_train, X_test, y_train):
    # Creates the vectorizer object
    vectorizer = CountVectorizer()
    # Vectorizes and storages the data
    X_train_v = vectorizer.fit_transform(X_train)
    # Vectorizes the data
    X_test_v = vectorizer.transform(X_test)

    # Creates and trains the model
    model = MultinomialNB()
    model.fit(X_train_v, y_train)

    return model, X_test_v


def model_prediction(model, X_test_v):
    # Predicts over the vectorized data
    y_pred = model.predict(X_test_v)
    return y_pred


def create_print_dataframe(X_test, y_test, y_pred):
    # Creates a new dataframe and gives it a format to print it
    df = pd.DataFrame()
    df['Mensaje'] = X_test
    df['Etiqueta'] = y_test
    df['Predicciones'] = y_pred
    print(df.to_string(index=False, justify='center', col_space=20))


def evaluate_model(y_test, y_pred):
    positiveClass = 'spam'
    negativeClass = 'no spam'
    accuracy = accuracy_score(y_test, y_pred)
    # Uses pos_label to denote which one is the positive class (1)
    f1 = f1_score(y_test, y_pred, pos_label=positiveClass)
    precision = precision_score(y_test, y_pred, pos_label=positiveClass)
    recall = recall_score(y_test, y_pred, pos_label=positiveClass)
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[negativeClass, positiveClass])
    # Separates the elements of the matrix
    tn, fp, fn, tp = confusionMatrix.ravel()
    specificity = tn / (tn + fp)
    print("---Metrics---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}\n")
    ConfusionMatrixDisplay(confusionMatrix, display_labels=['Not Spam', 'Spam']).plot()
    plt.title("Confusion Matrix")
    plt.show()


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=450)
    model, X_test_v = model_training(X_train, X_test, y_train)
    y_pred = model_prediction(model, X_test_v)
    create_print_dataframe(X_test, y_test, y_pred)
    evaluate_model(y_test, y_pred)


if __name__ == '__main__':
    main()
