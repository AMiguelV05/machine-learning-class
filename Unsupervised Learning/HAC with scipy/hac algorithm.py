from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Reading the dataset
def load_data():
    df = pd.read_csv('iris_plant (2).csv')
    return df

# Main logic
def linkage_and_visualization(df, max_clusters):
    # Assign the characteristics to a variable X
    X = df.drop('iris', axis=1)
    # Normalize X
    X_normalized = StandardScaler().fit_transform(X)
    # Define the methods we're using for the linkage
    methods = ["single", "complete", "average"]
    # Calculate the linkage for each method
    for method in methods:
        result = linkage(X_normalized, method, 'euclidean')
        visualization(df, X_normalized, result, method, max_clusters)


def visualization(df, X_normalized, result, method, max_clusters):
    # Calculate the cutting distance for the dendrogram on (max_clusters)
    if len(result) >= max_clusters:
        cut_distance = (result[-max_clusters, 2] + result[-(max_clusters - 1), 2]) / 2
    else:
        cut_distance = 0
    # Create the dendrogram, set the labels and draw the cut line
    dn = dendrogram(result)
    plt.title(f'Dendrogram using {method} linkage')
    plt.axhline(y=cut_distance, linestyle='--', color='r', label=f'Cut for {max_clusters} clusters')
    plt.legend()
    plt.show()
    # Get the clusters from result, whit max: max_clusters = 3
    labels = fcluster(result, max_clusters, criterion='maxclust')
    # For each cluster, we set the points on the chart
    for i in range(1, max_clusters + 1):
        plt.scatter(X_normalized[labels == i, 0], X_normalized[labels == i, 1], label=f'Cluster {i}')
    # Set title, labels, and showing the chart
    plt.title(f'Scatter chart using {method} linkage')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend()
    plt.show()
    # Create a new DataFrame with clusters and the actual iris classes
    comparison = pd.DataFrame({'Clusters': labels, 'Actual': df['iris']})
    # Create a numPy array of zeros of object type with size = labels
    map_prediction = np.zeros_like(labels, dtype=object)
    # For each cluster
    for i in range(1, max_clusters + 1):
        # Get the iris mode from the clusters
        majority_class = comparison[comparison['Clusters'] == i]['Actual'].mode()[0]
        # Replace the 0's to the name of the resultant classes from the previous calculation
        map_prediction[labels == i] = majority_class

    # Get the classes we have on the dataset
    real_classes = df['iris'].unique()
    # Create the confusion matrix with the real classes and the predictions
    confusionM = confusion_matrix(df['iris'], map_prediction)
    display_matrix = ConfusionMatrixDisplay(confusionM, display_labels=real_classes)
    display_matrix.plot()
    plt.title(f'Confusion Matrix for {method} linkage')
    plt.show()


def main():
    max_clusters = 3
    df = load_data()
    linkage_and_visualization(df, max_clusters)


if __name__ == "__main__":
    main()
