from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import time

# Step 1: Load the dataset and extract embeddings
print("Loading dataset...")
dataset = load_dataset("tonyassi/vogue-runway-top15-512px-nobg-embeddings2")["train"]
embeddings = np.array([item["embeddings"] for item in dataset])
print(f"Total embeddings loaded: {len(embeddings)}")

# Step 2: Define function to run benchmarks
def run_clustering_benchmark(embeddings, sample_sizes=[1000, 1000, 3000], pca_components=150):
    results = []

    for size in sample_sizes:
        print(f"\n--- Sample size: {size} ---")
        X = embeddings[:size]
        X = StandardScaler().fit_transform(X)

        # PCA
        pca = PCA(n_components=pca_components)
        X_pca = pca.fit_transform(X)

        for name, algo in [
            ("KMeans", KMeans(n_clusters=400, random_state=42)),
            ("DBSCAN", DBSCAN(eps=0.5, min_samples=5)),
            ("BIRCH", Birch(threshold=0.5, n_clusters=400))
        ]:
            try:
                start = time.time()
                labels = algo.fit_predict(X_pca)
                end = time.time()

                runtime = round(end - start, 4)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                sil_score = round(silhouette_score(X_pca, labels), 4) if n_clusters > 1 else "N/A"

                print(f"{name}: Time = {runtime}s | Clusters = {n_clusters} | Silhouette = {sil_score}")

                results.append({
                    "Algorithm": name,
                    "Samples": size,
                    "Runtime (s)": runtime,
                    "Clusters Found": n_clusters,
                    "Silhouette Score": sil_score
                })

            except Exception as e:
                print(f"{name} failed: {str(e)}")
                results.append({
                    "Algorithm": name,
                    "Samples": size,
                    "Runtime (s)": "Failed",
                    "Clusters Found": "Failed",
                    "Silhouette Score": "Failed"
                })

    return pd.DataFrame(results)

# Step 3: Run benchmark and save results
benchmark_df = run_clustering_benchmark(embeddings)
print("\n=== Final Results ===")
print(benchmark_df)

benchmark_df.to_csv("clustering_benchmark_results.csv", index=False)
