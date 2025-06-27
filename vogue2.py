from datasets import load_dataset
from collections import Counter
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import Birch
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


class fashionTrendClusterer:
    def __init__(self, dataset_name="tonyassi/vogue-runway-top15-512px-nobg-embeddings2", top_k_clusters=10):
        self.dataset_name = dataset_name
        self.top_k_clusters = top_k_clusters
        self.dataset = None
        self.embeddings = None
        self.labels = None
        self.seasons = []
        self.years = []
        self.cluster_labels = None
        self.reduced_2d = None
        self.df = None
        self.top_clusters = None
        self.rep_indices = []
        self.image_paths = [] #option

#LOAD VOGUE DATASET FROM HUGGINGFACE
    def load_dataset(self):
        dataset = load_dataset(self.dataset_name)["train"]
        label_mapping = dataset.features["label"].int2str
        dataset = dataset.filter(lambda example: "menswear" not in label_mapping(example["label"]).lower())
        self.dataset = dataset
        self.embeddings = np.array([item["embeddings"] for item in dataset])
        self.labels = [label_mapping(item["label"]) for item in dataset]

#EXTRACT LABELS DESIGNER, SEASON, YEAR
        pattern = r'(\b(?:pre\s)?\w+\b)\s(\d{4})'
        for label in self.labels:
            match = re.search(pattern, label)
            if match:
                self.seasons.append(match.group(1))
                self.years.append(match.group(2))

#BIRCH CLUSTERING PARAMETERS 
    def cluster_embeddings(self, threshold=12):
        pca = PCA(n_components=150)
        embeddings_pca = pca.fit_transform(self.embeddings)
        explained_variance = sum(pca.explained_variance_ratio_)
        print(f"Cumulative Explained Variance Ratio (PCA): {explained_variance:.2f}")

        birch = Birch(branching_factor=50, threshold=threshold, n_clusters=None)
        self.cluster_labels = birch.fit_predict(embeddings_pca)

#calc silohuttte score
        silhouette_avg = silhouette_score(embeddings_pca, self.cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.2f}")
        print(len(set(self.cluster_labels)))


        cluster_counts = Counter(self.cluster_labels)
        self.top_clusters = [c[0] for c in cluster_counts.most_common(self.top_k_clusters)]

        self.reduced_2d = PCA(n_components=2).fit_transform(embeddings_pca)
        self.df = pd.DataFrame({
            "x": self.reduced_2d[:, 0],
            "y": self.reduced_2d[:, 1],
            "Cluster": self.cluster_labels,
            "Designer": [l.split("-")[0].strip() for l in self.labels],
            "Season": self.seasons,
            "Year": self.years
        })


    def visualize_clusters(self, interactive=True):
        """plot cluster scatter graph using plot.ly- dash"""


        #dataframe labels
        df_top = self.df[self.df["Cluster"].isin(self.top_clusters)].copy()
        df_top["Label"] = df_top["Designer"] + " - " + df_top["Season"] + " " + df_top["Year"]

        # Compute 2D centroids in PCA space
        centroids_2d = []
        for cluster_id in self.top_clusters:
            cluster_points = self.reduced_2d[self.df["Cluster"] == cluster_id]
            centroid_2d = cluster_points.mean(axis=0)
            centroids_2d.append({
                "x": centroid_2d[0],
                "y": centroid_2d[1],
                "Cluster": str(cluster_id)
            })
        centroids_df = pd.DataFrame(centroids_2d)

    #scatter graph that plots tha pca points of the top ten clusters
        if interactive:
            fig = px.scatter(
                df_top, x="x", y="y", color=df_top["Cluster"].astype(str),
                hover_name="Label", title="Top 10 BIRCH Trends from 1995 - 2024",
                width=1200, height=850
            )
            fig.update_layout(
                paper_bgcolor="#FAF6F7",
                plot_bgcolor="#FFFFFF"
)

            return fig.to_html(full_html=False)
        else:
            return None
        

    def get_trend_dataframe(self):
        """
        creates a dataframe of year, season, cluster and 
        returns this dataframe that is for the top ten clusters 
        where the frequecny and year is scaled for the trend forecaster
        """
        trend_df = pd.DataFrame({
            "Year": [int(y) for y in self.years],
            "Season": self.seasons,
            "Cluster": self.cluster_labels
        })

        trend_df = trend_df.groupby(["Year", "Season", "Cluster"]).size().reset_index(name="Frequency")

        top_clusters = trend_df.groupby("Cluster")["Frequency"].sum().nlargest(10).index
        top_clusters_df = trend_df[trend_df["Cluster"].isin(top_clusters)].copy()

        # Save original values for plotting
        top_clusters_df["OriginalYear"] = top_clusters_df["Year"]
        top_clusters_df["OriginalFrequency"] = top_clusters_df["Frequency"]

        # Scale both Year and Frequency
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(top_clusters_df[["Year", "Frequency"]])
        top_clusters_df["Year"] = scaled_values[:, 0]
        top_clusters_df["ScaledFrequency"] = scaled_values[:, 1]

        return top_clusters_df

    def get_designer_dataframe(self):
        """
        Returns a DataFrame of Designer, Season, and Year along with the corresponding embeddings.
        """
        designer_names = [label.split("-")[0].strip() for label in self.labels]
        years_int = [int(y) for y in self.years]

        meta_df = pd.DataFrame({
            "Designer": designer_names,
            "Season": self.seasons,
            "Year": years_int
        })

        return meta_df, self.embeddings
    
    def find_representative_images(self):
        """
        For each of the top 10 clusters, find the two images closest to the cluster centroid.
        Save their DataFrame indices in a dictionary: {cluster_id: [idx1, idx2]}.
        """
        self.rep_indices = {}

        for cluster_id in self.top_clusters:
            # Filter the DataFrame for current cluster
            cluster_df = self.df[self.df['Cluster'] == cluster_id].copy()

            # Get indices and corresponding embeddings
            cluster_indices = cluster_df.index.tolist()
            cluster_embeddings = self.embeddings[cluster_indices]

            # Get centroid of the cluster
            centroid = np.mean(cluster_embeddings, axis=0)

            # Compute Euclidean distance from each point to the centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

            # Get indices of 2 closest points
            closest_indices = np.argsort(distances)[:2]

            # Save original DataFrame indices
            self.rep_indices[cluster_id] = [cluster_indices[i] for i in closest_indices]







