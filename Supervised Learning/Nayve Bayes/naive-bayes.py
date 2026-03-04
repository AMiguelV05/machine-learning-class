from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import pandas as pd
from matplotlib import pyplot as plt


def load_data():
    df = pd.read_excel("spam_dataset.xlsx")
    X = df['Mensaje']
    y = df['Etiqueta']
    return X, y

def model_training(X_train, X_test, y_train):
    vectorizer = CountVectorizer()
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_v, y_train)

    return model, X_test_v


def model_prediction(model, X_test_v):
    y_pred = model.predict(X_test_v)
    return y_pred


def create_print_dataframe(X_test, y_test, y_pred):
    df = pd.DataFrame()
    df['Mensaje'] = X_test
    df['Etiqueta'] = y_test
    df['Predicciones'] = y_pred
    print(df.to_string(index=False, justify='center', col_space=20))


def evaluate_model(y_test, y_pred):
    positiveClass = 'spam'
    negativeClass = 'no spam'
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=positiveClass)
    precision = precision_score(y_test, y_pred, pos_label=positiveClass)
    recall = recall_score(y_test, y_pred, pos_label=positiveClass)
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[negativeClass, positiveClass])
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
