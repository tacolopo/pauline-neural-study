"""
Visualization Module
====================

Publication-quality visualizations for the Pauline Neural Study.

This module creates a comprehensive suite of plots for exploring and
presenting results from bootstrap-stabilized word embeddings, cross-epistle
analysis, and semantic clustering. All plots use consistent styling and
are designed for both interactive exploration and inclusion in academic
publications.

Plot Types
----------
    Embedding Space Visualizations
        2D projections of the high-dimensional embedding space using
        t-SNE or UMAP. Theological terms are highlighted with distinct
        markers and labels, enabling visual inspection of semantic
        neighborhoods.

    Stability Heatmaps
        Color-coded matrices showing neighbor stability or pair stability
        across bootstrap samples. Hot cells indicate robust relationships;
        cool cells indicate unstable ones.

    Cross-Epistle Influence Charts
        Bar and heatmap charts showing which epistles most influence
        specific word relationships. Reveals epistle-specific theological
        vocabulary patterns.

    Word Relationship Networks
        Graph visualizations where nodes are words and edges represent
        semantic similarity above a configurable threshold. Edge width
        and color encode similarity and stability, respectively.

    Semantic Field Cluster Plots
        Scatter plots of semantic clusters with convex hull boundaries,
        annotated with cluster labels and theological terms.

    Bootstrap Distribution Plots
        Histograms and violin plots showing the distribution of similarity
        scores across bootstrap samples for specific word pairs. Useful
        for assessing the precision of semantic distance estimates.

Configuration
-------------
    All plots are saved to a configurable output directory. File format,
    DPI, and figure dimensions are controlled via constructor parameters.
    The default style uses a clean, academic aesthetic based on seaborn's
    ``whitegrid`` theme.

Dependencies
------------
    - matplotlib >= 3.7
    - seaborn >= 0.12
    - numpy
    - scikit-learn (for t-SNE / optional UMAP)
    - networkx (for relationship network graphs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from numpy.typing import NDArray

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI use
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize
import seaborn as sns

from ..embeddings.trainer import EmbeddingResult
from ..cross_epistle.analyzer import CrossEpistleResult, WordRelationship
from ..bootstrap.sampler import BootstrapResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlotConfig:
    """Configuration for plot aesthetics and output.

    Attributes:
        figsize: Default figure size as (width, height) in inches.
        dpi: Resolution for saved figures. Use 300+ for publication.
        file_format: Output file format ('png', 'pdf', 'svg', 'eps').
        style: Seaborn style preset. One of 'whitegrid', 'darkgrid',
            'white', 'dark', 'ticks'.
        palette: Seaborn color palette name. 'deep', 'muted', 'pastel',
            'bright', 'dark', 'colorblind' are good defaults.
        font_scale: Scaling factor for all font sizes.
        context: Seaborn context preset controlling element sizes.
            One of 'paper', 'notebook', 'talk', 'poster'.
        title_fontsize: Font size for plot titles.
        label_fontsize: Font size for axis labels.
        annotation_fontsize: Font size for text annotations on plots.
        theological_marker: Marker symbol for theological terms.
        theological_color: Color for theological term markers.
        highlight_alpha: Alpha (transparency) for highlighted regions.
    """
    figsize: tuple[float, float] = (12, 8)
    dpi: int = 300
    file_format: str = "png"
    style: str = "whitegrid"
    palette: str = "deep"
    font_scale: float = 1.2
    context: str = "paper"
    title_fontsize: int = 16
    label_fontsize: int = 13
    annotation_fontsize: int = 9
    theological_marker: str = "*"
    theological_color: str = "#D32F2F"
    highlight_alpha: float = 0.3


# ---------------------------------------------------------------------------
# Main visualizer class
# ---------------------------------------------------------------------------

class Visualizer:
    """
    Visualization engine for Pauline Neural Study results.

    Creates, styles, and saves all plot types. Each ``plot_*`` method
    produces a matplotlib figure, saves it to the output directory, and
    optionally returns the figure for further customization.

    Parameters
    ----------
    output_dir : str or Path
        Directory where all plots will be saved. Created automatically
        if it does not exist.
    config : PlotConfig, optional
        Styling and output configuration. Uses sensible academic defaults
        if not provided.

    Examples
    --------
    >>> viz = Visualizer(output_dir="./figures")
    >>> viz.plot_embedding_space(embedding_result, highlight_words=["faith", "law"])
    >>> viz.plot_stability_heatmap(embedding_result, terms=contested_terms)
    >>> viz.plot_bootstrap_distributions(embedding_result, word_pairs=[("faith", "grace")])
    """

    def __init__(
        self,
        output_dir: str | Path = "./figures",
        config: Optional[PlotConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or PlotConfig()

        # Apply global style settings
        sns.set_theme(
            style=self.config.style,
            palette=self.config.palette,
            font_scale=self.config.font_scale,
            context=self.config.context,
        )

        logger.info(f"Visualizer initialized, output directory: {self.output_dir}")

    def _save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        tight_layout: bool = True,
    ) -> Path:
        """Save a figure to the output directory.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        filename : str
            Base filename without extension (extension added from config).
        tight_layout : bool
            Whether to apply tight_layout before saving.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        if tight_layout:
            fig.tight_layout()

        filepath = self.output_dir / f"{filename}.{self.config.file_format}"
        fig.savefig(
            filepath,
            dpi=self.config.dpi,
            format=self.config.file_format,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info(f"Saved plot: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Embedding space visualization (t-SNE / UMAP)
    # ------------------------------------------------------------------

    def plot_embedding_space(
        self,
        embedding_result: EmbeddingResult,
        method: Literal["tsne", "umap"] = "tsne",
        highlight_words: Optional[list[str]] = None,
        label_top_n: int = 30,
        perplexity: float = 30.0,
        random_state: int = 42,
        filename: str = "embedding_space",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a 2D projection of the embedding space.

        Projects the high-dimensional mean embeddings into 2D using
        t-SNE or UMAP, then plots all vocabulary words as scatter points.
        Theological terms and user-specified highlight words are marked
        with distinct colors and labeled.

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Embedding data to visualize.
        method : {'tsne', 'umap'}
            Dimensionality reduction method. t-SNE preserves local
            structure (neighborhoods); UMAP preserves both local and
            global structure.
        highlight_words : list[str], optional
            Words to highlight with special markers and labels.
        label_top_n : int
            Number of most-stable words to label on the plot (in addition
            to any highlight words).
        perplexity : float
            t-SNE perplexity parameter. Higher values consider more
            neighbors, producing more global structure. Typical range: 5-50.
        random_state : int
            Random seed for reproducible projections.
        filename : str
            Output filename (without extension).
        title : str, optional
            Custom plot title. Defaults to a descriptive auto-generated title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure (also saved to output directory).
        """
        embeddings = embedding_result.mean_embeddings
        vocabulary = embedding_result.vocabulary

        # Dimensionality reduction
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=2,
                perplexity=min(perplexity, len(vocabulary) - 1),
                random_state=random_state,
                max_iter=1000,
            )
            coords_2d = reducer.fit_transform(embeddings)
        elif method == "umap":
            try:
                import umap
            except ImportError:
                raise ImportError(
                    "UMAP requires the 'umap-learn' package. "
                    "Install with: pip install umap-learn"
                )
            reducer = umap.UMAP(
                n_components=2,
                random_state=random_state,
                n_neighbors=15,
                min_dist=0.1,
            )
            coords_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown projection method: {method}. Use 'tsne' or 'umap'.")

        # Determine which words to label
        highlight_set = set(highlight_words or [])
        stability_ranked = sorted(
            embedding_result.neighbor_stability.items(),
            key=lambda x: -x[1],
        )
        top_stable = {w for w, _ in stability_ranked[:label_top_n]}
        label_words = highlight_set | top_stable

        # Build color array based on stability scores
        stability_values = np.array([
            embedding_result.neighbor_stability.get(w, 0.0)
            for w in vocabulary
        ])
        norm = Normalize(vmin=0, vmax=1)

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Plot all points colored by stability
        scatter = ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=stability_values,
            cmap="viridis",
            s=15,
            alpha=0.6,
            norm=norm,
            edgecolors="none",
        )

        # Highlight theological / user-specified words
        for i, word in enumerate(vocabulary):
            if word in highlight_set:
                ax.scatter(
                    coords_2d[i, 0],
                    coords_2d[i, 1],
                    marker=self.config.theological_marker,
                    c=self.config.theological_color,
                    s=200,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.5,
                )

            if word in label_words:
                fontweight = "bold" if word in highlight_set else "normal"
                fontsize = self.config.annotation_fontsize + (2 if word in highlight_set else 0)
                ax.annotate(
                    word,
                    (coords_2d[i, 0], coords_2d[i, 1]),
                    fontsize=fontsize,
                    fontweight=fontweight,
                    alpha=0.85,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Neighbor Stability", fontsize=self.config.label_fontsize)

        # Labels and title
        method_label = "t-SNE" if method == "tsne" else "UMAP"
        if title is None:
            title = (
                f"Pauline Embedding Space ({method_label} Projection)\n"
                f"{len(vocabulary)} words, {embedding_result.n_samples} bootstrap samples"
            )
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_xlabel(f"{method_label} Dimension 1", fontsize=self.config.label_fontsize)
        ax.set_ylabel(f"{method_label} Dimension 2", fontsize=self.config.label_fontsize)

        self._save_figure(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Stability heatmap
    # ------------------------------------------------------------------

    def plot_stability_heatmap(
        self,
        embedding_result: EmbeddingResult,
        terms: Optional[list[str]] = None,
        metric: Literal["similarity", "stability"] = "stability",
        filename: str = "stability_heatmap",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a heatmap of pairwise similarity or stability between terms.

        Displays a symmetric matrix where each cell shows either the
        cosine similarity or the bootstrap stability of the relationship
        between two words. Useful for identifying which theological term
        pairs have robust vs. fragile semantic relationships.

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Embedding data with stability metrics.
        terms : list[str], optional
            Terms to include. Defaults to the most stable words.
        metric : {'similarity', 'stability'}
            Which metric to display in the heatmap cells.
        filename : str
            Output filename (without extension).
        title : str, optional
            Custom title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated heatmap figure.
        """
        if terms is None:
            # Use top 20 most-stable words
            ranked = sorted(
                embedding_result.neighbor_stability.items(),
                key=lambda x: -x[1],
            )
            terms = [w for w, _ in ranked[:20]]

        # Filter to terms actually in vocabulary
        terms = [t for t in terms if t in embedding_result.word_to_idx]
        n = len(terms)

        if n < 2:
            logger.warning("Need at least 2 terms for heatmap")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Insufficient terms for heatmap", ha="center", va="center")
            self._save_figure(fig, filename)
            return fig

        # Build matrix
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    if metric == "similarity":
                        matrix[i, j] = embedding_result.similarity(terms[i], terms[j])
                    else:  # stability
                        pair_key = tuple(sorted([terms[i], terms[j]]))
                        matrix[i, j] = embedding_result.pair_stability.get(pair_key, 0.0)

        # Create figure
        fig_width = max(8, n * 0.6 + 3)
        fig_height = max(6, n * 0.5 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        cmap = "YlOrRd" if metric == "stability" else "RdBu_r"
        vmin = 0.0 if metric == "stability" else -1.0
        vmax = 1.0

        heatmap = sns.heatmap(
            matrix,
            xticklabels=terms,
            yticklabels=terms,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": metric.capitalize(), "shrink": 0.8},
            ax=ax,
        )

        # Rotate labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        if title is None:
            metric_label = "Pair Stability" if metric == "stability" else "Cosine Similarity"
            title = (
                f"Pauline Word {metric_label}\n"
                f"{embedding_result.n_samples} bootstrap samples"
            )
        ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)

        self._save_figure(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Cross-epistle influence chart
    # ------------------------------------------------------------------

    def plot_cross_epistle_influence(
        self,
        cross_epistle_result: CrossEpistleResult,
        top_n_pairs: int = 15,
        filename: str = "cross_epistle_influence",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a chart showing which epistles most influence word relationships.

        Produces a grouped bar chart where each group is a word pair and
        each bar shows the influence score of a specific epistle on that
        pair's similarity. High influence means removing that epistle
        significantly changes the word pair's relationship.

        Parameters
        ----------
        cross_epistle_result : CrossEpistleResult
            Cross-epistle analysis output with influence scores.
        top_n_pairs : int
            Number of most-influenced word pairs to display.
        filename : str
            Output filename.
        title : str, optional
            Custom title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        # Collect all influence data into a flat structure
        rows: list[dict] = []
        for epistle, pair_influences in cross_epistle_result.epistle_influence.items():
            for pair_key, influence in pair_influences.items():
                rows.append({
                    "epistle": epistle,
                    "pair": pair_key.replace(":", " -- "),
                    "influence": influence,
                })

        if not rows:
            logger.warning("No cross-epistle influence data to plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No influence data available", ha="center", va="center")
            self._save_figure(fig, filename)
            return fig

        # Find top influenced pairs (by maximum influence across all epistles)
        pair_max_influence: dict[str, float] = {}
        for row in rows:
            pair = row["pair"]
            pair_max_influence[pair] = max(
                pair_max_influence.get(pair, 0.0),
                row["influence"],
            )

        top_pairs = sorted(pair_max_influence.items(), key=lambda x: -x[1])[:top_n_pairs]
        top_pair_set = {pair for pair, _ in top_pairs}
        filtered_rows = [r for r in rows if r["pair"] in top_pair_set]

        # Build matrix: pairs x epistles
        epistles = sorted(cross_epistle_result.epistle_influence.keys())
        pairs = [pair for pair, _ in top_pairs]

        influence_matrix = np.zeros((len(pairs), len(epistles)))
        for row in filtered_rows:
            if row["pair"] in pairs and row["epistle"] in epistles:
                i = pairs.index(row["pair"])
                j = epistles.index(row["epistle"])
                influence_matrix[i, j] = row["influence"]

        # Plot as heatmap
        fig_height = max(6, len(pairs) * 0.5 + 2)
        fig_width = max(8, len(epistles) * 0.8 + 3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.heatmap(
            influence_matrix,
            xticklabels=epistles,
            yticklabels=pairs,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Influence Score", "shrink": 0.8},
            ax=ax,
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel("Epistle Removed", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Word Pair", fontsize=self.config.label_fontsize)

        if title is None:
            title = (
                f"Cross-Epistle Influence on Word Relationships\n"
                f"{cross_epistle_result.epistle_subsets_analyzed} subsets analyzed"
            )
        ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)

        self._save_figure(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Word relationship network
    # ------------------------------------------------------------------

    def plot_word_network(
        self,
        embedding_result: EmbeddingResult,
        words: Optional[list[str]] = None,
        min_similarity: float = 0.3,
        min_stability: float = 0.3,
        filename: str = "word_network",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a network graph of word relationships.

        Nodes represent words; edges connect words whose cosine similarity
        exceeds ``min_similarity``. Edge width encodes similarity strength
        and edge color encodes bootstrap stability.

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Embedding data with stability metrics.
        words : list[str], optional
            Subset of words to include. Defaults to the top 40 most-stable
            words in the vocabulary.
        min_similarity : float
            Minimum cosine similarity for an edge to be drawn.
        min_stability : float
            Minimum pair stability for an edge to be drawn. Edges below
            this threshold are omitted even if similarity is high.
        filename : str
            Output filename.
        title : str, optional
            Custom title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated network figure.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "Network visualization requires the 'networkx' package. "
                "Install with: pip install networkx"
            )

        if words is None:
            ranked = sorted(
                embedding_result.neighbor_stability.items(),
                key=lambda x: -x[1],
            )
            words = [w for w, _ in ranked[:40]]

        words = [w for w in words if w in embedding_result.word_to_idx]

        # Build graph
        G = nx.Graph()
        for word in words:
            stability = embedding_result.neighbor_stability.get(word, 0.0)
            G.add_node(word, stability=stability)

        edge_similarities: list[float] = []
        edge_stabilities: list[float] = []

        for i, w1 in enumerate(words):
            for w2 in words[i + 1:]:
                sim = embedding_result.similarity(w1, w2)
                pair_key = tuple(sorted([w1, w2]))
                stab = embedding_result.pair_stability.get(pair_key, 0.0)

                if sim >= min_similarity and stab >= min_stability:
                    G.add_edge(w1, w2, similarity=sim, stability=stab)
                    edge_similarities.append(sim)
                    edge_stabilities.append(stab)

        if len(G.edges) == 0:
            logger.warning(
                "No edges pass the similarity/stability thresholds. "
                "Try lowering min_similarity or min_stability."
            )

        # Layout
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(len(G.nodes)))
        else:
            pos = {}

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Draw edges with width proportional to similarity
        if edge_similarities:
            sim_array = np.array(edge_similarities)
            stab_array = np.array(edge_stabilities)
            # Normalize widths to [0.5, 4.0]
            width_min, width_max = 0.5, 4.0
            if sim_array.max() > sim_array.min():
                widths = width_min + (sim_array - sim_array.min()) / (sim_array.max() - sim_array.min()) * (width_max - width_min)
            else:
                widths = np.full_like(sim_array, (width_min + width_max) / 2)

            # Edge color from stability
            edge_cmap = matplotlib.colormaps["RdYlGn"]
            edge_colors = [edge_cmap(s) for s in stab_array]

            edges = list(G.edges())
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                width=widths.tolist(),
                edge_color=edge_colors,
                alpha=0.7,
                ax=ax,
            )

        # Draw nodes with size proportional to stability
        node_stabilities = [G.nodes[n].get("stability", 0.5) for n in G.nodes]
        node_sizes = [300 + s * 700 for s in node_stabilities]

        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_stabilities,
            cmap="viridis",
            vmin=0,
            vmax=1,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            ax=ax,
        )

        # Labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=self.config.annotation_fontsize,
            font_weight="bold",
            ax=ax,
        )

        # Add colorbars for edges and nodes
        sm_nodes = cm.ScalarMappable(cmap="viridis", norm=Normalize(0, 1))
        sm_nodes.set_array([])
        cbar_nodes = fig.colorbar(sm_nodes, ax=ax, shrink=0.5, pad=0.02, location="right")
        cbar_nodes.set_label("Node Stability", fontsize=self.config.annotation_fontsize)

        if edge_similarities:
            sm_edges = cm.ScalarMappable(cmap="RdYlGn", norm=Normalize(0, 1))
            sm_edges.set_array([])
            cbar_edges = fig.colorbar(sm_edges, ax=ax, shrink=0.5, pad=0.06, location="right")
            cbar_edges.set_label("Edge Stability", fontsize=self.config.annotation_fontsize)

        ax.set_axis_off()

        if title is None:
            title = (
                f"Pauline Word Relationship Network\n"
                f"{len(G.nodes)} words, {len(G.edges)} connections "
                f"(sim >= {min_similarity}, stab >= {min_stability})"
            )
        ax.set_title(title, fontsize=self.config.title_fontsize)

        self._save_figure(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Semantic field cluster plot
    # ------------------------------------------------------------------

    def plot_semantic_clusters(
        self,
        embedding_result: EmbeddingResult,
        cluster_labels: dict[str, int],
        method: Literal["tsne", "umap"] = "tsne",
        cluster_names: Optional[dict[int, str]] = None,
        perplexity: float = 30.0,
        random_state: int = 42,
        filename: str = "semantic_clusters",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a 2D scatter plot with semantic cluster coloring and hulls.

        Projects words into 2D (t-SNE or UMAP), colors them by cluster
        membership, and draws convex hull boundaries around each cluster.
        Cluster labels and theological terms are annotated.

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Embedding data.
        cluster_labels : dict[str, int]
            Maps each word to its cluster ID (from SemanticAnalyzer).
        method : {'tsne', 'umap'}
            Projection method.
        cluster_names : dict[int, str], optional
            Human-readable names for each cluster ID.
        perplexity : float
            t-SNE perplexity.
        random_state : int
            Random seed.
        filename : str
            Output filename.
        title : str, optional
            Custom title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated cluster plot.
        """
        from scipy.spatial import ConvexHull

        # Filter to words in both vocabulary and cluster_labels
        words = [
            w for w in embedding_result.vocabulary
            if w in cluster_labels and w in embedding_result.word_to_idx
        ]

        if len(words) < 3:
            logger.warning("Need at least 3 clustered words for cluster plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Insufficient data for cluster plot", ha="center", va="center")
            self._save_figure(fig, filename)
            return fig

        indices = [embedding_result.word_to_idx[w] for w in words]
        vectors = embedding_result.mean_embeddings[indices]
        labels = [cluster_labels[w] for w in words]

        # Dimensionality reduction
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=2,
                perplexity=min(perplexity, len(words) - 1),
                random_state=random_state,
                max_iter=1000,
            )
            coords_2d = reducer.fit_transform(vectors)
        else:
            try:
                import umap
            except ImportError:
                raise ImportError(
                    "UMAP requires the 'umap-learn' package. "
                    "Install with: pip install umap-learn"
                )
            reducer = umap.UMAP(
                n_components=2,
                random_state=random_state,
                n_neighbors=15,
                min_dist=0.1,
            )
            coords_2d = reducer.fit_transform(vectors)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        unique_labels = sorted(set(labels))
        n_clusters = len(unique_labels)
        colors = sns.color_palette(self.config.palette, n_colors=max(n_clusters, 1))

        for ci, cluster_id in enumerate(unique_labels):
            cluster_mask = [i for i, l in enumerate(labels) if l == cluster_id]
            cluster_coords = coords_2d[cluster_mask]
            cluster_words = [words[i] for i in cluster_mask]

            color = colors[ci % len(colors)]
            name = (cluster_names or {}).get(cluster_id, f"Cluster {cluster_id}")

            # Scatter points
            ax.scatter(
                cluster_coords[:, 0],
                cluster_coords[:, 1],
                c=[color],
                s=60,
                alpha=0.8,
                label=name,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

            # Convex hull (if >= 3 points)
            if len(cluster_coords) >= 3:
                try:
                    hull = ConvexHull(cluster_coords)
                    hull_points = np.append(hull.vertices, hull.vertices[0])
                    ax.fill(
                        cluster_coords[hull_points, 0],
                        cluster_coords[hull_points, 1],
                        color=color,
                        alpha=self.config.highlight_alpha * 0.5,
                        zorder=1,
                    )
                    ax.plot(
                        cluster_coords[hull_points, 0],
                        cluster_coords[hull_points, 1],
                        color=color,
                        alpha=0.6,
                        linewidth=1.5,
                        linestyle="--",
                        zorder=2,
                    )
                except Exception:
                    # ConvexHull can fail for degenerate cases
                    pass

            # Label words
            for i, word in zip(cluster_mask, cluster_words):
                ax.annotate(
                    word,
                    (coords_2d[i, 0], coords_2d[i, 1]),
                    fontsize=self.config.annotation_fontsize,
                    alpha=0.8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

        ax.legend(
            loc="best",
            fontsize=self.config.annotation_fontsize,
            framealpha=0.9,
            title="Semantic Clusters",
        )

        method_label = "t-SNE" if method == "tsne" else "UMAP"
        if title is None:
            title = (
                f"Pauline Semantic Clusters ({method_label})\n"
                f"{len(words)} words in {n_clusters} clusters"
            )
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_xlabel(f"{method_label} Dimension 1", fontsize=self.config.label_fontsize)
        ax.set_ylabel(f"{method_label} Dimension 2", fontsize=self.config.label_fontsize)

        self._save_figure(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Bootstrap distribution plots
    # ------------------------------------------------------------------

    def plot_bootstrap_distributions(
        self,
        embedding_result: EmbeddingResult,
        word_pairs: list[tuple[str, str]],
        filename: str = "bootstrap_distributions",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the distribution of similarity scores across bootstrap samples.

        For each specified word pair, shows a violin plot (or histogram)
        of the cosine similarity computed in each individual bootstrap
        sample. This visualizes the precision of our semantic distance
        estimates -- tight distributions indicate robust relationships.

        Requires that ``embedding_result.sample_embeddings`` is not None
        (i.e., the embedding trainer was run with ``store_all_embeddings=True``).

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Embedding data with per-sample embeddings.
        word_pairs : list[tuple[str, str]]
            Word pairs to plot distributions for.
        filename : str
            Output filename.
        title : str, optional
            Custom title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated distribution plot.

        Raises
        ------
        ValueError
            If per-sample embeddings are not available.
        """
        if embedding_result.sample_embeddings is None:
            raise ValueError(
                "Bootstrap distribution plots require per-sample embeddings. "
                "Re-run the embedding trainer with store_all_embeddings=True."
            )

        # Filter to valid pairs
        valid_pairs = [
            (w1, w2) for w1, w2 in word_pairs
            if w1 in embedding_result.word_to_idx and w2 in embedding_result.word_to_idx
        ]

        if not valid_pairs:
            logger.warning("No valid word pairs for distribution plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No valid word pairs", ha="center", va="center")
            self._save_figure(fig, filename)
            return fig

        # Compute per-sample similarities for each pair
        from scipy.spatial.distance import cosine as cosine_dist

        pair_distributions: dict[str, list[float]] = {}
        for w1, w2 in valid_pairs:
            idx1 = embedding_result.word_to_idx[w1]
            idx2 = embedding_result.word_to_idx[w2]
            pair_label = f"{w1} -- {w2}"
            sims = []
            for sample_emb in embedding_result.sample_embeddings:
                sim = 1.0 - cosine_dist(sample_emb[idx1], sample_emb[idx2])
                sims.append(sim)
            pair_distributions[pair_label] = sims

        # Create figure with violin plots
        n_pairs = len(pair_distributions)
        fig_height = max(6, n_pairs * 1.2 + 2)
        fig, ax = plt.subplots(figsize=(self.config.figsize[0], fig_height))

        pair_labels = list(pair_distributions.keys())
        data_for_plot = [pair_distributions[label] for label in pair_labels]

        # Violin plot
        parts = ax.violinplot(
            data_for_plot,
            positions=range(len(pair_labels)),
            vert=False,
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )

        # Style the violins
        for pc in parts["bodies"]:
            pc.set_facecolor(sns.color_palette(self.config.palette)[0])
            pc.set_alpha(0.7)

        # Overlay individual points (jittered)
        rng = np.random.default_rng(42)
        for i, data in enumerate(data_for_plot):
            jitter = rng.normal(0, 0.04, size=len(data))
            ax.scatter(
                data,
                np.full(len(data), i) + jitter,
                s=3,
                alpha=0.3,
                color="black",
                zorder=2,
            )

        # Add mean similarity text
        for i, (label, data) in enumerate(zip(pair_labels, data_for_plot)):
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.text(
                max(data) + 0.02,
                i,
                f"mean={mean_val:.3f} (sd={std_val:.3f})",
                fontsize=self.config.annotation_fontsize,
                va="center",
            )

        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels, fontsize=self.config.annotation_fontsize + 1)
        ax.set_xlabel("Cosine Similarity", fontsize=self.config.label_fontsize)

        if title is None:
            title = (
                f"Bootstrap Similarity Distributions\n"
                f"{embedding_result.n_samples} bootstrap samples"
            )
        ax.set_title(title, fontsize=self.config.title_fontsize)

        self._save_figure(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Stability across bootstrap samples (per-word bar chart)
    # ------------------------------------------------------------------

    def plot_stability_ranking(
        self,
        embedding_result: EmbeddingResult,
        top_n: int = 40,
        highlight_words: Optional[list[str]] = None,
        filename: str = "stability_ranking",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Bar chart ranking words by their neighbor stability score.

        Shows the top-N most stable words, optionally highlighting
        specific theological terms. Stability is the average Jaccard
        similarity of a word's nearest-neighbor set across bootstrap
        samples.

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Embedding data with neighbor stability scores.
        top_n : int
            Number of top words to display.
        highlight_words : list[str], optional
            Words to highlight with a distinct bar color.
        filename : str
            Output filename.
        title : str, optional
            Custom title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated bar chart.
        """
        highlight_set = set(highlight_words or [])

        ranked = sorted(
            embedding_result.neighbor_stability.items(),
            key=lambda x: -x[1],
        )[:top_n]

        words = [w for w, _ in ranked]
        scores = [s for _, s in ranked]
        colors = [
            self.config.theological_color if w in highlight_set
            else sns.color_palette(self.config.palette)[0]
            for w in words
        ]

        fig_height = max(6, top_n * 0.35 + 2)
        fig, ax = plt.subplots(figsize=(self.config.figsize[0], fig_height))

        y_pos = range(len(words))
        ax.barh(y_pos, scores, color=colors, edgecolor="white", linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=self.config.annotation_fontsize + 1)
        ax.set_xlabel("Neighbor Stability (Jaccard)", fontsize=self.config.label_fontsize)
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()  # Highest stability at top

        # Add value annotations
        for i, (word, score) in enumerate(zip(words, scores)):
            ax.text(
                score + 0.01,
                i,
                f"{score:.3f}",
                va="center",
                fontsize=self.config.annotation_fontsize,
            )

        if title is None:
            title = (
                f"Word Stability Ranking (Top {top_n})\n"
                f"{embedding_result.n_samples} bootstrap samples"
            )
        ax.set_title(title, fontsize=self.config.title_fontsize)

        # Add legend if there are highlighted words
        if highlight_set:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=sns.color_palette(self.config.palette)[0], label="General vocabulary"),
                Patch(facecolor=self.config.theological_color, label="Theological terms"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=self.config.annotation_fontsize)

        self._save_figure(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Cross-epistle similarity variation
    # ------------------------------------------------------------------

    def plot_epistle_similarity_variation(
        self,
        cross_epistle_result: CrossEpistleResult,
        word_pairs: Optional[list[tuple[str, str]]] = None,
        top_n: int = 10,
        filename: str = "epistle_similarity_variation",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Show how word-pair similarity varies across epistle subsets.

        For each word pair, draws a horizontal strip chart showing the
        distribution of similarity scores across different epistle
        subset configurations. The baseline (all epistles) is marked
        with a vertical line.

        Parameters
        ----------
        cross_epistle_result : CrossEpistleResult
            Cross-epistle analysis output.
        word_pairs : list[tuple[str, str]], optional
            Specific pairs to plot. If None, selects the most variable.
        top_n : int
            Number of word pairs to display if ``word_pairs`` is None.
        filename : str
            Output filename.
        title : str, optional
            Custom title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated variation plot.
        """
        relationships = cross_epistle_result.relationships

        if word_pairs is None:
            # Select the most variable relationships
            variable = sorted(
                relationships.values(),
                key=lambda r: -r.range,
            )[:top_n]
        else:
            variable = [
                relationships.get((w1, w2), relationships.get((w2, w1)))
                for w1, w2 in word_pairs
            ]
            variable = [r for r in variable if r is not None]

        if not variable:
            logger.warning("No word pair variation data to plot")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No variation data", ha="center", va="center")
            self._save_figure(fig, filename)
            return fig

        fig_height = max(6, len(variable) * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(self.config.figsize[0], fig_height))

        for i, rel in enumerate(variable):
            pair_label = f"{rel.word1} -- {rel.word2}"
            sims = list(rel.similarities.values())
            baseline = rel.similarities.get("all", rel.mean_similarity)

            # Strip chart
            ax.scatter(
                sims,
                [i] * len(sims),
                s=25,
                alpha=0.6,
                color=sns.color_palette(self.config.palette)[0],
                zorder=3,
            )

            # Baseline marker
            ax.axvline(x=baseline, color="gray", linestyle=":", alpha=0.3)
            ax.scatter(
                [baseline],
                [i],
                marker="|",
                s=200,
                color=self.config.theological_color,
                linewidths=2,
                zorder=4,
                label="Full corpus" if i == 0 else None,
            )

            # Range line
            ax.plot(
                [min(sims), max(sims)],
                [i, i],
                color="gray",
                linewidth=1,
                alpha=0.5,
                zorder=2,
            )

        pair_labels = [f"{r.word1} -- {r.word2}" for r in variable]
        ax.set_yticks(range(len(variable)))
        ax.set_yticklabels(pair_labels, fontsize=self.config.annotation_fontsize + 1)
        ax.set_xlabel("Cosine Similarity", fontsize=self.config.label_fontsize)
        ax.invert_yaxis()
        ax.legend(loc="lower right", fontsize=self.config.annotation_fontsize)

        if title is None:
            title = (
                f"Word Pair Similarity Variation Across Epistle Subsets\n"
                f"{cross_epistle_result.epistle_subsets_analyzed} configurations"
            )
        ax.set_title(title, fontsize=self.config.title_fontsize)

        self._save_figure(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Semantic distance matrix visualization
    # ------------------------------------------------------------------

    def plot_distance_matrix(
        self,
        terms: list[str],
        distances: NDArray,
        similarities: NDArray,
        stability: Optional[NDArray] = None,
        filename: str = "distance_matrix",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize a semantic distance matrix as annotated heatmaps.

        Creates a figure with one or two panels: the left panel shows
        cosine similarity (warm colors = close), and the right panel
        (if stability data is available) shows pair stability.

        Parameters
        ----------
        terms : list[str]
            Term labels for the matrix axes.
        distances : NDArray
            Square distance matrix.
        similarities : NDArray
            Square similarity matrix.
        stability : NDArray, optional
            Square stability matrix.
        filename : str
            Output filename.
        title : str, optional
            Custom title.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        n_panels = 2 if stability is not None else 1
        fig_width = self.config.figsize[0] * n_panels * 0.6 + 2
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(fig_width, self.config.figsize[1]),
        )
        if n_panels == 1:
            axes = [axes]

        # Panel 1: Similarity
        sns.heatmap(
            similarities,
            xticklabels=terms,
            yticklabels=terms,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
            ax=axes[0],
        )
        axes[0].set_title("Semantic Similarity", fontsize=self.config.label_fontsize)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

        # Panel 2: Stability (if available)
        if stability is not None:
            sns.heatmap(
                stability,
                xticklabels=terms,
                yticklabels=terms,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                square=True,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "Pair Stability", "shrink": 0.8},
                ax=axes[1],
            )
            axes[1].set_title("Bootstrap Stability", fontsize=self.config.label_fontsize)
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

        if title is None:
            title = f"Semantic Distance Analysis ({len(terms)} theological terms)"
        fig.suptitle(title, fontsize=self.config.title_fontsize, y=1.02)

        self._save_figure(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Combined summary figure
    # ------------------------------------------------------------------

    def plot_summary_dashboard(
        self,
        embedding_result: EmbeddingResult,
        cross_epistle_result: Optional[CrossEpistleResult] = None,
        highlight_words: Optional[list[str]] = None,
        filename: str = "summary_dashboard",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a multi-panel summary dashboard.

        Combines four key visualizations into a single figure:

        1. **Top-left**: Embedding space (t-SNE projection)
        2. **Top-right**: Stability ranking (top 20 words)
        3. **Bottom-left**: Stability heatmap (top 15 word pairs)
        4. **Bottom-right**: Cross-epistle variation (top 8 pairs)
           or a second stability view if no cross-epistle data is available

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Embedding data.
        cross_epistle_result : CrossEpistleResult, optional
            Cross-epistle data for the bottom-right panel.
        highlight_words : list[str], optional
            Words to highlight in the embedding space panel.
        filename : str
            Output filename.
        title : str, optional
            Overall dashboard title.

        Returns
        -------
        matplotlib.figure.Figure
            The multi-panel summary figure.
        """
        fig = plt.figure(figsize=(20, 16))

        # Panel 1: Embedding space (t-SNE) -- top-left
        ax1 = fig.add_subplot(2, 2, 1)
        self._mini_embedding_space(ax1, embedding_result, highlight_words)

        # Panel 2: Stability ranking -- top-right
        ax2 = fig.add_subplot(2, 2, 2)
        self._mini_stability_ranking(ax2, embedding_result, highlight_words)

        # Panel 3: Stability heatmap -- bottom-left
        ax3 = fig.add_subplot(2, 2, 3)
        self._mini_stability_heatmap(ax3, embedding_result)

        # Panel 4: Cross-epistle variation or extra info -- bottom-right
        ax4 = fig.add_subplot(2, 2, 4)
        if cross_epistle_result is not None:
            self._mini_epistle_variation(ax4, cross_epistle_result)
        else:
            self._mini_pair_stability_heatmap(ax4, embedding_result)

        if title is None:
            title = (
                f"Pauline Neural Study Summary\n"
                f"{len(embedding_result.vocabulary)} words, "
                f"{embedding_result.n_samples} bootstrap samples, "
                f"dim={embedding_result.embedding_dim}"
            )
        fig.suptitle(title, fontsize=self.config.title_fontsize + 2, y=1.01)

        self._save_figure(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Mini-panel helper methods (for dashboard)
    # ------------------------------------------------------------------

    def _mini_embedding_space(
        self,
        ax: plt.Axes,
        embedding_result: EmbeddingResult,
        highlight_words: Optional[list[str]] = None,
    ) -> None:
        """Draw a compact embedding space projection on the given axes."""
        from sklearn.manifold import TSNE

        embeddings = embedding_result.mean_embeddings
        vocabulary = embedding_result.vocabulary

        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(vocabulary) - 1),
            random_state=42,
            max_iter=500,
        )
        coords_2d = reducer.fit_transform(embeddings)

        stability_values = np.array([
            embedding_result.neighbor_stability.get(w, 0.0)
            for w in vocabulary
        ])

        ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=stability_values, cmap="viridis", s=10, alpha=0.5,
            vmin=0, vmax=1,
        )

        highlight_set = set(highlight_words or [])
        for i, word in enumerate(vocabulary):
            if word in highlight_set:
                ax.scatter(
                    coords_2d[i, 0], coords_2d[i, 1],
                    marker=self.config.theological_marker,
                    c=self.config.theological_color,
                    s=100, zorder=5,
                )
                ax.annotate(
                    word, (coords_2d[i, 0], coords_2d[i, 1]),
                    fontsize=7, fontweight="bold",
                    xytext=(3, 3), textcoords="offset points",
                )

        ax.set_title("Embedding Space (t-SNE)", fontsize=self.config.label_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    def _mini_stability_ranking(
        self,
        ax: plt.Axes,
        embedding_result: EmbeddingResult,
        highlight_words: Optional[list[str]] = None,
    ) -> None:
        """Draw a compact stability ranking on the given axes."""
        highlight_set = set(highlight_words or [])
        ranked = sorted(
            embedding_result.neighbor_stability.items(),
            key=lambda x: -x[1],
        )[:20]

        words = [w for w, _ in ranked]
        scores = [s for _, s in ranked]
        colors = [
            self.config.theological_color if w in highlight_set
            else sns.color_palette(self.config.palette)[0]
            for w in words
        ]

        ax.barh(range(len(words)), scores, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        ax.set_title("Stability Ranking (Top 20)", fontsize=self.config.label_fontsize)
        ax.set_xlabel("Stability", fontsize=9)

    def _mini_stability_heatmap(
        self,
        ax: plt.Axes,
        embedding_result: EmbeddingResult,
    ) -> None:
        """Draw a compact stability heatmap on the given axes."""
        ranked = sorted(
            embedding_result.neighbor_stability.items(),
            key=lambda x: -x[1],
        )
        terms = [w for w, _ in ranked[:15]]

        n = len(terms)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = embedding_result.similarity(terms[i], terms[j])

        sns.heatmap(
            matrix, xticklabels=terms, yticklabels=terms,
            cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.3,
            cbar_kws={"shrink": 0.6},
            ax=ax, annot=False,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
        ax.set_title("Similarity (Top 15 Stable Words)", fontsize=self.config.label_fontsize)

    def _mini_epistle_variation(
        self,
        ax: plt.Axes,
        cross_epistle_result: CrossEpistleResult,
    ) -> None:
        """Draw a compact epistle variation chart on the given axes."""
        variable = sorted(
            cross_epistle_result.relationships.values(),
            key=lambda r: -r.range,
        )[:8]

        for i, rel in enumerate(variable):
            sims = list(rel.similarities.values())
            baseline = rel.similarities.get("all", rel.mean_similarity)

            ax.scatter(sims, [i] * len(sims), s=12, alpha=0.5,
                       color=sns.color_palette(self.config.palette)[0])
            ax.scatter([baseline], [i], marker="|", s=100,
                       color=self.config.theological_color, linewidths=1.5)
            ax.plot([min(sims), max(sims)], [i, i],
                    color="gray", linewidth=0.5, alpha=0.4)

        pair_labels = [f"{r.word1}-{r.word2}" for r in variable]
        ax.set_yticks(range(len(variable)))
        ax.set_yticklabels(pair_labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Similarity", fontsize=9)
        ax.set_title("Epistle Variation (Top 8 Pairs)", fontsize=self.config.label_fontsize)

    def _mini_pair_stability_heatmap(
        self,
        ax: plt.Axes,
        embedding_result: EmbeddingResult,
    ) -> None:
        """Draw a compact pair stability heatmap on the given axes."""
        ranked = sorted(
            embedding_result.neighbor_stability.items(),
            key=lambda x: -x[1],
        )
        terms = [w for w, _ in ranked[:15]]

        n = len(terms)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    pair_key = tuple(sorted([terms[i], terms[j]]))
                    matrix[i, j] = embedding_result.pair_stability.get(pair_key, 0.0)

        sns.heatmap(
            matrix, xticklabels=terms, yticklabels=terms,
            cmap="YlOrRd", vmin=0, vmax=1,
            square=True, linewidths=0.3,
            cbar_kws={"shrink": 0.6},
            ax=ax, annot=False,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
        ax.set_title("Pair Stability (Top 15 Words)", fontsize=self.config.label_fontsize)
