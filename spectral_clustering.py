from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import pandas as pd


def spectral_clustering(input_file_name, output_file_name):
    # Reading of csv-file
    df = pd.read_csv(input_file_name)

    # Choosing of important columns
    X = df[['Clicks', 'Impressions']].dropna().copy()

    # Creation of Spectral Clustering model
    spectral_cluster_model = SpectralClustering(
        n_clusters=5,
        random_state=64,
        n_neighbors=20,
        affinity='nearest_neighbors'
    )

    # Cluster prediction
    X['cluster'] = spectral_cluster_model.fit_predict(X.values)
    labels = X['cluster']

    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
    plt.title('Spectral Clustering')
    plt.show()

    df['cluster'] = X['cluster']
    df.to_csv(output_file_name, index=False)
