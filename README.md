# Protein Clustering Analysis

This is a data analysis project built with Python to perform clustering on protein sequences. The pipeline processes sequences from a FASTA file, extracts k-mer based features, applies dimensionality reduction, and compares multiple clustering algorithms to find an optimal grouping.

## Workflow

The main analysis script (`protein_clustering.py`) executes the following steps:

1.  **Process FASTA**: Reads a specified FASTA file (e.g., `astral-scopdom-seqres-gd-sel-95-2.08.fa`), extracting all protein sequences and their corresponding class labels.
2.  **Feature Extraction**: For each sequence, it extracts "2x2" k-mers (4-mers with a skip) and builds a high-dimensional, sparse binary feature matrix where each row is a sequence and each column is a unique k-mer pattern.
3.  **Dimensionality Reduction**: Applies **TruncatedSVD** (a form of PCA for sparse matrices) to reduce the feature matrix to a manageable number of components (e.g., 300).
4.  **Algorithm Comparison**: Runs a suite of clustering algorithms on the reduced data, including **KMeans**, **DBSCAN**, **OPTICS**, **Agglomerative Clustering**, **Birch**, and more.
5.  **Metric Evaluation**: Evaluates each algorithm's performance using:
      * **Internal Metrics** (unsupervised): Silhouette Score, Calinski-Harabasz Score, Davies-Bouldin Score.
      * **External Metrics** (supervised, using true labels): F1-Macro Score, Adjusted Rand Score (ARI), Normalized Mutual Info Score (NMI).
6.  **Optimization**: Selects the best-performing algorithm (based on a chosen internal metric like 'silhouette') and performs a grid search to find the optimal hyperparameters.
7.  **Visualization**: A separate script (`cluster_distribution.py`) can be run after the main analysis to load the final cluster labels and generate a bar chart showing the distribution of the largest clusters.

## Project Scripts

  * `protein_clustering.py`: The main analysis pipeline. This script handles data loading, feature engineering, PCA, algorithm comparison, and hyperparameter optimization.
  * `cluster_distribution.py`: A utility script to visualize the final cluster assignments saved by the main script. It plots a bar chart of the Top 20 clusters, plus any "Noise" (cluster -1) and an "Outros" (Others) category.
  * `README.md`: The project documentation file (this file).

## Requirements

The project relies on the following main Python libraries:

  * `numpy`
  * `pandas`
  * `scikit-learn` (sklearn)
  * `matplotlib`
  * `seaborn`
  * `tqdm`

## How to Run

1.  **Run the Main Analysis**:
    Ensure your FASTA file (e.g., `astral-scopdom-seqres-gd-sel-95-2.08.fa`) is in the same directory. Then, execute the main script:

    ```bash
    python protein_clustering.py
    ```

2.  **Visualize the Results**:
    After the main script finishes (and `best_clustering_labels.npy` is created), run the visualization script:

    ```bash
    python cluster_distribution.py
    ```
## Outputs

This pipeline will generate the following files:

  * `clustering_results.csv`: A table comparing the internal and external evaluation metrics for all tested clustering algorithms.
  * `pca_variance.png`: A line plot showing the cumulative explained variance by the number of SVD components.
 
<div align="center"> 
     <img src="https://github.com/xyz2094/protein-clustering/blob/main/outputs/pca_variance.png" width=480 height="360">
</div>

  * `best_clustering_labels.npy`: A NumPy array file containing the final predicted cluster labels from the best-optimized model.
  * `top20_cluster_distribution_graph`: A bar chart visualizing the size of the top 20 clusters, saved by `cluster_distribution.py`.
  
<div align="center"> 
     <img src="https://github.com/xyz2094/protein-clustering/blob/main/outputs/grafico_distribuicao_clusters_TOP20.png" width=480 height="360">
</div>
