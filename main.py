import pandas as pd
from data_normalization import main_normalization
from spectral_clustering import spectral_clustering
from lda import make_lda
from date_analysis import date_analysis

df = pd.read_csv("FacebookAds.csv")

df = main_normalization(df)

df.to_csv("cleaned_FacebookAds.csv", index=False)

spectral_clustering("cleaned_FacebookAds.csv", "ads_with_clusters.csv")

make_lda("ads_with_clusters.csv", "ads_with_clusters_and_lda.csv")

date_analysis("ads_with_clusters.csv")
