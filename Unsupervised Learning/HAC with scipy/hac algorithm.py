from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

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
    plt.title(f'Dendrogram using {method}')
    plt.axhline(y=cut_distance, linestyle='--', color='r', label=f'Cut for {max_clusters} clusters')
    plt.legend()
    plt.show()
    # Get the clusters from result, whit max: max_clusters = 3
    labels = fcluster(result, max_clusters, criterion='maxclust')
    # For each cluster, we set the points on the chart
    for i in range(1, max_clusters + 1):
        plt.scatter(X_normalized[labels == i, 0], X_normalized[labels == i, 1], label=f'Cluster {i}')
    # Set title, labels, and showing the chart
    plt.title(f'Scatter chart using {method}')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend()
    plt.show()
    # Create a new DataFrame with clusters and the actual iris classes
    comparison = pd.DataFrame({'Clusters': labels, 'Actual': df['iris']})
    # Create the table from the columns of the previous DataFrame
    table = pd.crosstab(comparison['Clusters'], comparison['Actual'])
    print(f"\nCOMPARISON TABLE FOR THE METHOD {method.upper()}\n")
    print(table)



def main():
    max_clusters = 3
    df = load_data()
    linkage_and_visualization(df, max_clusters)


if __name__ == "__main__":
    main()
