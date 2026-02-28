import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os

# Ignore irrelevant warnings
warnings.filterwarnings('ignore')

# ===================== Critical Fix: Set Matplotlib Backend =====================
# Switch to system default backend to resolve PyCharm compatibility issues
plt.switch_backend('TkAgg')  # Recommended for Windows; use 'Agg' for Mac, 'Qt5Agg' for Linux
# ================================================================================

# Set font to support Chinese characters (prevent garbled text in plots)
plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei font for Chinese display
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# ===================== Configuration Parameters =====================
DATASET_NAME = '哪吒之魔童闹海的影评'  # Keep Chinese for file path compatibility
DATA_ROOT_PATH = str(Path(__file__).parent.parent.parent.absolute())
# Path to input Excel file containing word vectors (output from previous script)
vector_input_path = os.path.join(DATA_ROOT_PATH, f"data/results_for_unsupervised_methods/{DATASET_NAME}_Nouns_WordVectors.xlsx")
# Range of cluster numbers to test (2-20 recommended for short texts)
CLUSTER_NUMBER_RANGE = range(2, 21)

# ======================================================================

def load_and_preprocess_word_vectors():
    """
    Load word vectors from Excel and perform preprocessing (filtering + normalization)
    Returns:
        words (list): List of valid vocabulary words
        normalized_vectors (numpy.ndarray): L2-normalized word vectors (shape: [n_words, 300])
    """
    print(f"Loading word vector file: {vector_input_path}")
    df = pd.read_excel(vector_input_path)

    # Extract vocabulary and vectors from DataFrame
    words = df['Vocabulary'].tolist()
    vectors = df.iloc[:, 1:].values  # Vectors start from the second column

    # Filter out invalid vectors (containing NaN values)
    valid_indices = ~np.isnan(vectors).any(axis=1)
    words = [words[i] for i in range(len(words)) if valid_indices[i]]
    vectors = vectors[valid_indices]

    # L2 normalization to improve KMeans clustering performance
    normalized_vectors = normalize(vectors, norm='l2')

    print(f"Number of valid words: {len(words)}")
    return words, normalized_vectors


def determine_optimal_cluster_number(vectors, cluster_range):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score
    Args:
        vectors (numpy.ndarray): Normalized word vectors
        cluster_range (range): Range of cluster numbers to test
    Returns:
        inertia_values (list): Inertia values for each cluster number
        silhouette_scores (list): Silhouette scores for each cluster number
        optimal_cluster_num (int): Optimal number of clusters (max silhouette score)
    """
    # Store evaluation metrics
    inertia_values = []  # Lower = better fit; elbow point indicates optimal K
    silhouette_scores = []  # Higher (closer to 1) = better cluster separation

    print("\nEvaluating clustering performance for different cluster numbers...")
    for n_clusters in cluster_range:
        # Initialize and fit KMeans model
        kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init='auto'  # Automatically select number of initializations
        )
        cluster_labels = kmeans_model.fit_predict(vectors)

        # Record inertia (core metric for Elbow Method)
        inertia_values.append(kmeans_model.inertia_)

        # Calculate Silhouette Score (requires ≥2 clusters with ≥1 sample each)
        if len(np.unique(cluster_labels)) > 1:
            avg_silhouette_score = silhouette_score(vectors, cluster_labels)
            silhouette_scores.append(avg_silhouette_score)
            print(f"Number of clusters={n_clusters} | Inertia={kmeans_model.inertia_:.2f} | Silhouette Score={avg_silhouette_score:.4f}")
        else:
            silhouette_scores.append(0)
            print(f"Number of clusters={n_clusters} | Inertia={kmeans_model.inertia_:.2f} | Silhouette Score=0 (invalid clustering)")

    # Determine optimal cluster number (max silhouette score)
    optimal_cluster_num = cluster_range[np.argmax(silhouette_scores)]
    max_silhouette_score = max(silhouette_scores)
    print(f"\n[Recommended Optimal Cluster Number]: {optimal_cluster_num} (Highest Silhouette Score: {max_silhouette_score:.4f})")

    return inertia_values, silhouette_scores, optimal_cluster_num


def visualize_evaluation_metrics(cluster_range, inertia_values, silhouette_scores):
    """
    Visualize Elbow Method and Silhouette Score for optimal cluster number selection
    Args:
        cluster_range (range): Tested cluster numbers
        inertia_values (list): Inertia values for each cluster number
        silhouette_scores (list): Silhouette scores for each cluster number
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Elbow Method (Inertia)
    ax1.plot(cluster_range, inertia_values, 'o-', color='blue', linewidth=2)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('KMeans Elbow Method (Find K at Elbow Point)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Silhouette Score
    ax2.plot(cluster_range, silhouette_scores, 'o-', color='red', linewidth=2)
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('KMeans Silhouette Score (Higher = Better)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Annotate optimal cluster number
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = cluster_range[optimal_idx]
    optimal_score = silhouette_scores[optimal_idx]
    ax2.scatter(optimal_k, optimal_score, color='darkred', s=100, label=f'Optimal K={optimal_k}')
    ax2.legend()

    # Save plot (prioritize saving to avoid display errors)
    plt.tight_layout()
    plot_save_path = os.path.join(DATA_ROOT_PATH, f"data/results_for_unsupervised_methods/{DATASET_NAME}_KMeans_Optimal_Cluster_Analysis.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: {plot_save_path}")

    # Attempt to display plot (fallback to save-only if error occurs)
    try:
        plt.show()
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"Note: Plot display failed (compatibility issue), but plot was saved locally. Error: {error_msg}")


if __name__ == '__main__':
    # Step 1: Load and preprocess word vectors
    vocabulary_words, normalized_vectors = load_and_preprocess_word_vectors()

    # Step 2: Evaluate different cluster numbers to find optimal value
    inertia_list, silhouette_list, best_cluster_num = determine_optimal_cluster_number(normalized_vectors, CLUSTER_NUMBER_RANGE)

    # Step 3: Visualize evaluation results
    visualize_evaluation_metrics(CLUSTER_NUMBER_RANGE, inertia_list, silhouette_list)