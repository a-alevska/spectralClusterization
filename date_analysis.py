import pandas as pd
import matplotlib.pyplot as plt


def date_analysis(input_file_name):
    df = pd.read_csv(input_file_name)

    df['CreationDate'] = df['CreationDate'].str.replace(' PDT', '', regex=False)
    df['CreationDate'] = pd.to_datetime(df['CreationDate'], errors='coerce', format='%m/%d/%y %I:%M:%S %p')
    df['CreationDate'] = df['CreationDate'].dropna()
    df['CreationDate'] = df['CreationDate'].dt.tz_localize('UTC')

    df['cluster'] = df['cluster'].dropna()
    clusters = df['cluster'].unique()

    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} data points")

    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]
        cluster_data = cluster_data.dropna(subset=['CreationDate'])

        cluster_data['CreationDate'] = pd.to_datetime(cluster_data['CreationDate'])

        plt.scatter(cluster_data['CreationDate'], cluster_data['Clicks'], label=f'Cluster {cluster}')
        plt.title(f'Clicks Over Time - Cluster {cluster}')
        plt.xlabel('Creation Date')
        plt.ylabel('Clicks')
        plt.legend()
        plt.show()

    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]
        cluster_data = cluster_data.dropna(subset=['CreationDate'])

        cluster_data['CreationDate'] = pd.to_datetime(cluster_data['CreationDate'])

        plt.scatter(cluster_data['CreationDate'], cluster_data['Clicks'], label=f'Cluster {cluster}')

    plt.title('Clicks Over Time')
    plt.xlabel('Creation Date')
    plt.ylabel('Clicks')
    plt.legend()
    plt.show()
