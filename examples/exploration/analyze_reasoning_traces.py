#!/usr/bin/env python3
"""
Script to analyze reasoning traces and embeddings from large language models.
This script provides visualization and analysis tools for the outputs of
visualize_reasoning_traces.py.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from collections import Counter


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """
    Load results from a JSON file.
    
    Args:
        results_path: Path to the results JSON file
        
    Returns:
        List of result dictionaries
    """
    print(f"Loading results from {results_path}")
    with open(results_path, "r") as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} results")
    return results


def load_embeddings(embeddings_path: str) -> np.ndarray:
    """
    Load embeddings from a NumPy file.
    
    Args:
        embeddings_path: Path to the embeddings NumPy file
        
    Returns:
        Array of embeddings
    """
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    print(f"Loaded embeddings with shape {embeddings.shape}")
    return embeddings


def visualize_embeddings_tsne(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    perplexity: int = 30,
    n_components: int = 2,
    random_state: int = 42,
    output_path: Optional[str] = None,
    title: str = "t-SNE Visualization of Reasoning Traces",
):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Array of embeddings
        labels: Optional list of labels for coloring points
        perplexity: Perplexity parameter for t-SNE
        n_components: Number of components for t-SNE
        random_state: Random state for reproducibility
        output_path: Path to save the visualization
        title: Title for the visualization
    """
    print(f"Running t-SNE with perplexity={perplexity}, n_components={n_components}")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot with or without labels
    if labels is not None:
        # Get unique labels
        unique_labels = list(set(labels))
        
        # Create a colormap
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))
        
        # Plot each label with a different color
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[cmap(i)],
                label=label,
                alpha=0.7,
            )
        
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    plt.title(title)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> List[int]:
    """
    Cluster embeddings using K-means.
    
    Args:
        embeddings: Array of embeddings
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        List of cluster labels
    """
    print(f"Clustering embeddings into {n_clusters} clusters")
    
    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Count samples per cluster
    cluster_counts = Counter(cluster_labels)
    for cluster_id, count in sorted(cluster_counts.items()):
        print(f"Cluster {cluster_id}: {count} samples")
    
    return cluster_labels


def analyze_reasoning_length(results: List[Dict[str, Any]]) -> None:
    """
    Analyze the length of reasoning traces.
    
    Args:
        results: List of result dictionaries
    """
    # Extract reasoning lengths
    reasoning_lengths = [len(result["reasoning"].split()) for result in results]
    
    # Calculate statistics
    avg_length = sum(reasoning_lengths) / len(reasoning_lengths)
    min_length = min(reasoning_lengths)
    max_length = max(reasoning_lengths)
    
    print(f"Reasoning length statistics:")
    print(f"  Average: {avg_length:.2f} words")
    print(f"  Minimum: {min_length} words")
    print(f"  Maximum: {max_length} words")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(reasoning_lengths, bins=20, alpha=0.7)
    plt.title("Distribution of Reasoning Trace Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def extract_common_phrases(
    texts: List[str],
    n_gram_range: tuple = (2, 4),
    top_n: int = 10,
) -> List[tuple]:
    """
    Extract common phrases from a list of texts.
    
    Args:
        texts: List of texts
        n_gram_range: Range of n-grams to consider
        top_n: Number of top phrases to return
        
    Returns:
        List of (phrase, count) tuples
    """
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Create vectorizer
    vectorizer = CountVectorizer(
        ngram_range=n_gram_range,
        stop_words="english",
        max_features=100,
    )
    
    # Fit and transform
    X = vectorizer.fit_transform(texts)
    
    # Get feature names and counts
    feature_names = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    top_phrases = [(feature_names[i], counts[i]) for i in sorted_indices[:top_n]]
    
    return top_phrases


def analyze_common_phrases(results: List[Dict[str, Any]], top_n: int = 10) -> None:
    """
    Analyze common phrases in reasoning traces.
    
    Args:
        results: List of result dictionaries
        top_n: Number of top phrases to show
    """
    # Extract reasoning texts
    reasoning_texts = [result["reasoning"] for result in results]
    
    # Extract common phrases
    common_phrases = extract_common_phrases(reasoning_texts, top_n=top_n)
    
    print(f"Top {top_n} common phrases in reasoning traces:")
    for phrase, count in common_phrases:
        print(f"  '{phrase}': {count} occurrences")


def analyze_clusters(
    results: List[Dict[str, Any]],
    cluster_labels: List[int],
    n_examples: int = 3,
) -> None:
    """
    Analyze clusters by showing examples from each cluster.
    
    Args:
        results: List of result dictionaries
        cluster_labels: List of cluster labels
        n_examples: Number of examples to show per cluster
    """
    # Group results by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(results[i])
    
    # Show examples from each cluster
    for cluster_id, cluster_results in sorted(clusters.items()):
        print(f"\nCluster {cluster_id} ({len(cluster_results)} samples):")
        
        # Extract common phrases for this cluster
        reasoning_texts = [result["reasoning"] for result in cluster_results]
        common_phrases = extract_common_phrases(reasoning_texts, top_n=5)
        
        print("  Common phrases:")
        for phrase, count in common_phrases:
            print(f"    '{phrase}': {count} occurrences")
        
        # Show examples
        print(f"  Examples (showing {min(n_examples, len(cluster_results))} of {len(cluster_results)}):")
        for i, result in enumerate(cluster_results[:n_examples]):
            print(f"    Example {i+1}:")
            print(f"      Query: {result['query']}")
            print(f"      Reasoning (truncated): {result['reasoning'][:100]}...")


def create_interactive_visualization(
    embeddings: np.ndarray,
    results: List[Dict[str, Any]],
    cluster_labels: Optional[List[int]] = None,
    output_path: str = "reasoning_visualization.html",
):
    """
    Create an interactive visualization of embeddings.
    
    Args:
        embeddings: Array of embeddings
        results: List of result dictionaries
        cluster_labels: Optional list of cluster labels
        output_path: Path to save the HTML file
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is required for interactive visualization. Install with: pip install plotly")
        return
    
    # Reduce dimensionality to 3D
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'query': [result['query'] for result in results],
        'reasoning': [result['reasoning'][:100] + '...' for result in results],
    })
    
    if cluster_labels is not None:
        df['cluster'] = [f"Cluster {label}" for label in cluster_labels]
        
        # Create figure with clusters
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='cluster',
            hover_data=['query', 'reasoning'],
            title="3D Visualization of Reasoning Traces",
        )
    else:
        # Create figure without clusters
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            hover_data=['query', 'reasoning'],
            title="3D Visualization of Reasoning Traces",
        )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            zaxis_title='PCA Component 3',
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    
    # Save as HTML
    fig.write_html(output_path)
    print(f"Interactive visualization saved to {output_path}")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Analyze reasoning traces and embeddings")
    
    # Input parameters
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to the results JSON file")
    parser.add_argument("--embeddings_path", type=str, required=True,
                        help="Path to the embeddings NumPy file")
    
    # Analysis parameters
    parser.add_argument("--analysis_type", type=str, default="all",
                        choices=["tsne", "clusters", "phrases", "length", "interactive", "all"],
                        help="Type of analysis to perform")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters for clustering analysis")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="Perplexity parameter for t-SNE")
    parser.add_argument("--top_n_phrases", type=int, default=10,
                        help="Number of top phrases to show")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./analysis_outputs",
                        help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    results = load_results(args.results_path)
    embeddings = load_embeddings(args.embeddings_path)
    
    # Perform analyses
    if args.analysis_type in ["tsne", "all"]:
        # t-SNE visualization
        output_path = os.path.join(args.output_dir, "tsne_visualization.png")
        visualize_embeddings_tsne(
            embeddings,
            perplexity=args.perplexity,
            output_path=output_path,
        )
    
    if args.analysis_type in ["clusters", "all"]:
        # Clustering
        cluster_labels = cluster_embeddings(embeddings, n_clusters=args.n_clusters)
        
        # Visualize with cluster labels
        output_path = os.path.join(args.output_dir, "tsne_clusters.png")
        visualize_embeddings_tsne(
            embeddings,
            labels=[f"Cluster {label}" for label in cluster_labels],
            perplexity=args.perplexity,
            output_path=output_path,
            title="t-SNE Visualization with Clusters",
        )
        
        # Analyze clusters
        analyze_clusters(results, cluster_labels)
    
    if args.analysis_type in ["phrases", "all"]:
        # Analyze common phrases
        analyze_common_phrases(results, top_n=args.top_n_phrases)
    
    if args.analysis_type in ["length", "all"]:
        # Analyze reasoning length
        analyze_reasoning_length(results)
    
    if args.analysis_type in ["interactive", "all"]:
        # Create interactive visualization
        if args.analysis_type == "all":
            # If running all analyses, we already have cluster labels
            cluster_labels = cluster_embeddings(embeddings, n_clusters=args.n_clusters)
        elif args.analysis_type == "interactive":
            # If only running interactive, we need to compute cluster labels
            cluster_labels = cluster_embeddings(embeddings, n_clusters=args.n_clusters)
        
        output_path = os.path.join(args.output_dir, "interactive_visualization.html")
        create_interactive_visualization(
            embeddings,
            results,
            cluster_labels=cluster_labels,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
