"""
Analysis Package
================

Semantic analysis and visualization tools for Pauline neural embedding studies.

This package provides two main interfaces:

    SemanticAnalyzer
        Analyzes contested theological terms in Paul's vocabulary using
        bootstrap-stabilized word embeddings. Builds semantic clusters,
        computes semantic distances between concepts, compares word usage
        across epistles, and generates semantic field maps.

    Visualizer
        Creates publication-quality visualizations of embedding spaces,
        stability metrics, cross-epistle influence patterns, and semantic
        field clusters. All plots are saveable to a configurable output
        directory.

Usage::

    from pauline.analysis import SemanticAnalyzer, Visualizer

    analyzer = SemanticAnalyzer(embedding_result)
    clusters = analyzer.build_semantic_clusters()
    distances = analyzer.compute_semantic_distances()

    viz = Visualizer(output_dir="./figures")
    viz.plot_embedding_space(embedding_result)
    viz.plot_stability_heatmap(embedding_result)
"""

from .semantic import SemanticAnalyzer
from .visualization import Visualizer

__all__ = [
    "SemanticAnalyzer",
    "Visualizer",
]
