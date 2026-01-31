"""
Semantic Analysis Tools
=======================

Analyzes Pauline word embeddings with a focus on contested theological
terminology. Paul's vocabulary contains terms that have been debated for
two millennia -- words whose meanings in Paul's own usage may differ from
how later theological traditions interpreted them.

By leveraging bootstrap-stabilized embeddings, this module identifies
Paul's *own* semantic structure: which concepts cluster together, which
are distant, and how usage varies across epistles.

Contested Theological Terms
---------------------------
The following Greek/English term pairs are central to Pauline theology
and are tracked as primary analysis targets:

    justification / dikaiosis (dikaiosyne, dikaioo)
        The act or state of being made righteous. Central to the
        Protestant-Catholic debate on faith vs. works.

    faith / pistis
        Trust, faithfulness, belief. The "pistis Christou" debate
        (faith IN Christ vs. faithfulness OF Christ) hinges on Paul's
        semantic usage of this term.

    law / nomos
        Torah, moral law, principle. Paul's complex and seemingly
        contradictory statements about nomos make it one of the most
        contested terms in Pauline studies.

    grace / charis
        Unmerited favor, gift. Closely linked to justification and
        faith in Paul's theological vocabulary.

Additional tracked terms include: spirit (pneuma), flesh (sarx),
righteousness (dikaiosyne), sin (hamartia), death (thanatos),
life (zoe), love (agape), hope (elpis), body (soma), cross (stauros).

Methods
-------
    build_semantic_clusters
        Groups words into semantic clusters based on embedding proximity,
        using agglomerative clustering with a configurable distance threshold.

    compare_across_epistles
        Measures how the semantic neighborhood of a word changes across
        different epistle subsets, identifying epistle-specific usage.

    generate_semantic_field_map
        Creates a structured mapping of semantic fields around a target
        word, organized by distance bands (near, mid, far).

    compute_semantic_distances
        Produces a distance matrix between all contested theological terms,
        using cosine distance in the mean embedding space.

    find_bridging_terms
        Identifies words that sit between two semantic clusters, potentially
        mediating Paul's conceptual transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from ..embeddings.trainer import EmbeddingResult
from ..cross_epistle.analyzer import CrossEpistleResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contested theological terms -- Greek transliterations and English glosses
# ---------------------------------------------------------------------------

#: Primary contested terms as (english, greek_transliteration) pairs.
CONTESTED_TERMS: list[tuple[str, str]] = [
    ("justification", "dikaiosis"),
    ("faith", "pistis"),
    ("law", "nomos"),
    ("grace", "charis"),
]

#: Extended theological vocabulary tracked in analysis.
EXTENDED_TERMS: list[tuple[str, str]] = [
    ("righteousness", "dikaiosyne"),
    ("spirit", "pneuma"),
    ("flesh", "sarx"),
    ("sin", "hamartia"),
    ("death", "thanatos"),
    ("life", "zoe"),
    ("love", "agape"),
    ("hope", "elpis"),
    ("body", "soma"),
    ("cross", "stauros"),
    ("circumcision", "peritome"),
    ("gospel", "euangelion"),
    ("salvation", "soteria"),
    ("redemption", "apolytrosis"),
    ("reconciliation", "katallage"),
    ("covenant", "diatheke"),
    ("promise", "epangelia"),
    ("works", "erga"),
    ("baptism", "baptisma"),
    ("resurrection", "anastasis"),
]

#: All tracked terms combined.
ALL_THEOLOGICAL_TERMS: list[tuple[str, str]] = CONTESTED_TERMS + EXTENDED_TERMS


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SemanticCluster:
    """A cluster of semantically related words discovered in the embedding space.

    Attributes:
        cluster_id: Integer identifier for this cluster.
        words: Words belonging to this cluster, sorted by proximity to centroid.
        centroid: Mean embedding vector of all words in the cluster.
        coherence: Average pairwise cosine similarity within the cluster.
            Values close to 1.0 indicate tightly grouped words; values near
            0.0 indicate a loose, heterogeneous grouping.
        theological_terms: Subset of ``words`` that appear in the tracked
            theological vocabulary.
        label: Optional human-readable label summarizing the cluster's
            semantic content (assigned after inspection).
    """
    cluster_id: int
    words: list[str]
    centroid: NDArray
    coherence: float
    theological_terms: list[str] = field(default_factory=list)
    label: Optional[str] = None

    def __repr__(self) -> str:
        term_str = ", ".join(self.words[:8])
        if len(self.words) > 8:
            term_str += f", ... ({len(self.words)} total)"
        return (
            f"SemanticCluster(id={self.cluster_id}, "
            f"coherence={self.coherence:.3f}, "
            f"words=[{term_str}])"
        )


@dataclass
class SemanticFieldMap:
    """A structured map of the semantic field surrounding a target word.

    Organizes the embedding neighborhood into concentric distance bands,
    giving a layered view of how meaning radiates outward from a concept.

    Attributes:
        target_word: The word at the center of the semantic field.
        near_field: Words within the closest distance band (highest similarity).
            These share the most direct semantic relationship with the target.
        mid_field: Words at moderate distance. Often related by broader
            thematic or syntactic patterns rather than direct synonymy.
        far_field: Words at the outer edge of the relevant neighborhood.
            These may share only loose topical association.
        stability_scores: Maps each neighbor word to its pair stability
            score with the target, indicating how robust the relationship
            is across bootstrap samples.
    """
    target_word: str
    near_field: list[tuple[str, float]]  # (word, similarity)
    mid_field: list[tuple[str, float]]
    far_field: list[tuple[str, float]]
    stability_scores: dict[str, float] = field(default_factory=dict)

    @property
    def all_neighbors(self) -> list[tuple[str, float]]:
        """All neighbors across all distance bands, sorted by similarity."""
        combined = self.near_field + self.mid_field + self.far_field
        return sorted(combined, key=lambda x: -x[1])

    @property
    def stable_near_field(self) -> list[tuple[str, float, float]]:
        """Near-field words annotated with stability scores.

        Returns:
            List of (word, similarity, stability) tuples.
        """
        return [
            (word, sim, self.stability_scores.get(word, 0.0))
            for word, sim in self.near_field
        ]


@dataclass
class SemanticDistanceMatrix:
    """Pairwise semantic distances between a set of theological terms.

    Attributes:
        terms: Ordered list of terms (row/column labels).
        distances: Square distance matrix (cosine distance, range [0, 2]).
        similarities: Square similarity matrix (1 - cosine distance).
        stability: Optional stability annotation for each pair.
    """
    terms: list[str]
    distances: NDArray  # shape: (n_terms, n_terms)
    similarities: NDArray  # shape: (n_terms, n_terms)
    stability: Optional[NDArray] = None  # shape: (n_terms, n_terms)

    def distance(self, term1: str, term2: str) -> float:
        """Look up the cosine distance between two terms."""
        if term1 not in self.terms or term2 not in self.terms:
            return float("nan")
        i = self.terms.index(term1)
        j = self.terms.index(term2)
        return float(self.distances[i, j])

    def similarity(self, term1: str, term2: str) -> float:
        """Look up the cosine similarity between two terms."""
        if term1 not in self.terms or term2 not in self.terms:
            return float("nan")
        i = self.terms.index(term1)
        j = self.terms.index(term2)
        return float(self.similarities[i, j])

    def closest_pair(self) -> tuple[str, str, float]:
        """Return the two terms with the smallest non-zero distance."""
        min_dist = float("inf")
        best_i, best_j = 0, 1
        n = len(self.terms)
        for i in range(n):
            for j in range(i + 1, n):
                if self.distances[i, j] < min_dist:
                    min_dist = self.distances[i, j]
                    best_i, best_j = i, j
        return self.terms[best_i], self.terms[best_j], float(min_dist)

    def farthest_pair(self) -> tuple[str, str, float]:
        """Return the two terms with the largest distance."""
        max_dist = -1.0
        best_i, best_j = 0, 1
        n = len(self.terms)
        for i in range(n):
            for j in range(i + 1, n):
                if self.distances[i, j] > max_dist:
                    max_dist = self.distances[i, j]
                    best_i, best_j = i, j
        return self.terms[best_i], self.terms[best_j], float(max_dist)


@dataclass
class EpistleComparisonResult:
    """Result of comparing a word's semantic neighborhood across epistles.

    Attributes:
        word: The word being compared.
        epistle_neighbors: Maps epistle subset label to the word's nearest
            neighbors in that subset's embedding space.
        neighbor_overlap: Jaccard similarity of neighbor sets between each
            pair of epistle subsets.
        most_variable_neighbor: The neighbor word whose rank changes most
            across epistle subsets.
        most_stable_neighbor: The neighbor word whose rank is most consistent.
    """
    word: str
    epistle_neighbors: dict[str, list[tuple[str, float]]]
    neighbor_overlap: dict[tuple[str, str], float]
    most_variable_neighbor: Optional[str] = None
    most_stable_neighbor: Optional[str] = None


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class SemanticAnalyzer:
    """
    Semantic analysis engine for bootstrap-stabilized Pauline word embeddings.

    Takes an ``EmbeddingResult`` (from the embedding trainer) and optionally
    a ``CrossEpistleResult`` (from the cross-epistle analyzer), then provides
    methods for investigating Paul's theological vocabulary structure.

    The analysis operates in the mean embedding space, which averages out
    the noise from individual bootstrap samples and surfaces the stable
    semantic relationships that persist across the corpus.

    Parameters
    ----------
    embedding_result : EmbeddingResult
        Bootstrap-aggregated embedding result containing the mean embedding
        matrix, vocabulary, and stability metrics.
    cross_epistle_result : CrossEpistleResult, optional
        Cross-epistle analysis result for epistle-level comparisons.
        If not provided, cross-epistle comparison methods will raise
        an error when called.
    theological_terms : list[tuple[str, str]], optional
        Override the default set of tracked theological terms. Each entry
        is an (english, greek_transliteration) pair. Defaults to
        ``ALL_THEOLOGICAL_TERMS``.

    Examples
    --------
    >>> analyzer = SemanticAnalyzer(embedding_result)
    >>> clusters = analyzer.build_semantic_clusters(n_clusters=8)
    >>> for cluster in clusters:
    ...     print(cluster.label, cluster.theological_terms)

    >>> field_map = analyzer.generate_semantic_field_map("faith")
    >>> for word, sim in field_map.near_field:
    ...     print(f"  {word}: {sim:.3f}")
    """

    def __init__(
        self,
        embedding_result: EmbeddingResult,
        cross_epistle_result: Optional[CrossEpistleResult] = None,
        theological_terms: Optional[list[tuple[str, str]]] = None,
    ):
        self.embeddings = embedding_result
        self.cross_epistle = cross_epistle_result
        self.theological_terms = theological_terms or ALL_THEOLOGICAL_TERMS

        # Build quick-lookup set of all tracked terms present in vocabulary
        self._tracked_terms: set[str] = set()
        for english, greek in self.theological_terms:
            if english in self.embeddings.word_to_idx:
                self._tracked_terms.add(english)
            if greek in self.embeddings.word_to_idx:
                self._tracked_terms.add(greek)

        logger.info(
            f"SemanticAnalyzer initialized: {len(self.embeddings.vocabulary)} vocabulary words, "
            f"{len(self._tracked_terms)} tracked theological terms found in embeddings"
        )

    # ------------------------------------------------------------------
    # Semantic clustering
    # ------------------------------------------------------------------

    def build_semantic_clusters(
        self,
        n_clusters: Optional[int] = None,
        distance_threshold: float = 0.7,
        method: str = "ward",
        words: Optional[list[str]] = None,
    ) -> list[SemanticCluster]:
        """
        Build semantic clusters from the embedding space.

        Uses agglomerative (hierarchical) clustering on the cosine distance
        matrix. Either specify a fixed number of clusters or a distance
        threshold for automatic cluster count determination.

        Parameters
        ----------
        n_clusters : int, optional
            Fixed number of clusters to produce. If provided, overrides
            ``distance_threshold``.
        distance_threshold : float
            Maximum cophenetic distance for merging clusters when
            ``n_clusters`` is not specified. Lower values produce more,
            tighter clusters. Range: [0, 2] for cosine distance.
        method : str
            Linkage method for hierarchical clustering. One of 'ward',
            'complete', 'average', 'single'. Ward linkage minimizes
            within-cluster variance and tends to produce balanced clusters.
        words : list[str], optional
            Subset of vocabulary words to cluster. If None, clusters all
            words in the vocabulary.

        Returns
        -------
        list[SemanticCluster]
            Clusters sorted by size (largest first), each annotated with
            coherence scores and identified theological terms.
        """
        if words is not None:
            target_words = [w for w in words if w in self.embeddings.word_to_idx]
        else:
            target_words = self.embeddings.vocabulary

        if len(target_words) < 2:
            logger.warning("Need at least 2 words for clustering")
            return []

        # Extract embedding vectors for target words
        indices = [self.embeddings.word_to_idx[w] for w in target_words]
        vectors = self.embeddings.mean_embeddings[indices]

        # Compute pairwise cosine distances
        dist_vector = pdist(vectors, metric="cosine")

        # Replace any NaN distances (from zero vectors) with maximum distance
        dist_vector = np.nan_to_num(dist_vector, nan=2.0)

        # Hierarchical clustering
        Z = linkage(dist_vector, method=method)

        if n_clusters is not None:
            labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        else:
            labels = fcluster(Z, t=distance_threshold, criterion="distance")

        # Organize words into clusters
        cluster_ids = set(labels)
        clusters: list[SemanticCluster] = []

        for cid in sorted(cluster_ids):
            member_indices = [i for i, label in enumerate(labels) if label == cid]
            member_words = [target_words[i] for i in member_indices]
            member_vectors = vectors[member_indices]

            # Compute centroid
            centroid = np.mean(member_vectors, axis=0)

            # Compute coherence (average pairwise cosine similarity)
            if len(member_indices) > 1:
                pairwise_dists = pdist(member_vectors, metric="cosine")
                coherence = float(1.0 - np.mean(pairwise_dists))
            else:
                coherence = 1.0

            # Sort words by proximity to centroid
            centroid_dists = [
                cosine(member_vectors[i], centroid)
                for i in range(len(member_indices))
            ]
            sorted_order = np.argsort(centroid_dists)
            member_words = [member_words[i] for i in sorted_order]

            # Identify theological terms in this cluster
            theo_terms = [w for w in member_words if w in self._tracked_terms]

            clusters.append(SemanticCluster(
                cluster_id=int(cid),
                words=member_words,
                centroid=centroid,
                coherence=coherence,
                theological_terms=theo_terms,
            ))

        # Sort clusters by size (largest first)
        clusters.sort(key=lambda c: -len(c.words))

        logger.info(
            f"Built {len(clusters)} semantic clusters from {len(target_words)} words "
            f"(coherence range: {min(c.coherence for c in clusters):.3f} - "
            f"{max(c.coherence for c in clusters):.3f})"
        )

        return clusters

    # ------------------------------------------------------------------
    # Cross-epistle comparison
    # ------------------------------------------------------------------

    def compare_across_epistles(
        self,
        word: str,
        top_n: int = 15,
    ) -> EpistleComparisonResult:
        """
        Compare a word's semantic neighborhood across epistle subsets.

        Requires a ``CrossEpistleResult`` to have been provided at
        initialization. For each epistle subset in the cross-epistle
        analysis, retrieves the word's neighbors and computes the overlap
        between neighbor sets.

        Parameters
        ----------
        word : str
            The word to compare across epistles.
        top_n : int
            Number of nearest neighbors to retrieve per subset.

        Returns
        -------
        EpistleComparisonResult
            Detailed comparison including neighbor lists per subset,
            pairwise neighbor overlap (Jaccard), and identification of
            the most stable and most variable neighbors.

        Raises
        ------
        ValueError
            If no ``CrossEpistleResult`` was provided at initialization.
        """
        if self.cross_epistle is None:
            raise ValueError(
                "Cross-epistle comparison requires a CrossEpistleResult. "
                "Provide one when initializing SemanticAnalyzer."
            )

        word = word.lower()
        if word not in self.embeddings.word_to_idx:
            raise ValueError(f"Word '{word}' not found in embedding vocabulary")

        # Gather neighbor data from cross-epistle relationships
        epistle_neighbors: dict[str, list[tuple[str, float]]] = {}

        # Use the cross-epistle result to find how relationships change
        for pair_key, relationship in self.cross_epistle.relationships.items():
            w1, w2 = pair_key
            if w1 != word and w2 != word:
                continue

            other_word = w2 if w1 == word else w1
            for subset_label, sim in relationship.similarities.items():
                if subset_label not in epistle_neighbors:
                    epistle_neighbors[subset_label] = []
                epistle_neighbors[subset_label].append((other_word, sim))

        # Sort each subset's neighbors by similarity
        for label in epistle_neighbors:
            epistle_neighbors[label].sort(key=lambda x: -x[1])
            epistle_neighbors[label] = epistle_neighbors[label][:top_n]

        # Compute pairwise Jaccard overlap of neighbor sets
        neighbor_overlap: dict[tuple[str, str], float] = {}
        labels = list(epistle_neighbors.keys())
        for i, label_a in enumerate(labels):
            set_a = {w for w, _ in epistle_neighbors[label_a]}
            for label_b in labels[i + 1:]:
                set_b = {w for w, _ in epistle_neighbors[label_b]}
                union = set_a | set_b
                intersection = set_a & set_b
                jaccard = len(intersection) / len(union) if union else 0.0
                neighbor_overlap[(label_a, label_b)] = jaccard

        # Find most stable and most variable neighbors
        # (appear across most / fewest subsets)
        neighbor_counts: dict[str, int] = {}
        for label, neighbors in epistle_neighbors.items():
            for other_word, _ in neighbors:
                neighbor_counts[other_word] = neighbor_counts.get(other_word, 0) + 1

        most_stable = max(neighbor_counts, key=neighbor_counts.get) if neighbor_counts else None
        most_variable = min(neighbor_counts, key=neighbor_counts.get) if neighbor_counts else None

        return EpistleComparisonResult(
            word=word,
            epistle_neighbors=epistle_neighbors,
            neighbor_overlap=neighbor_overlap,
            most_variable_neighbor=most_variable,
            most_stable_neighbor=most_stable,
        )

    # ------------------------------------------------------------------
    # Semantic field mapping
    # ------------------------------------------------------------------

    def generate_semantic_field_map(
        self,
        word: str,
        near_threshold: float = 0.7,
        mid_threshold: float = 0.4,
        max_neighbors: int = 50,
    ) -> SemanticFieldMap:
        """
        Generate a layered semantic field map around a target word.

        The embedding neighborhood is divided into three concentric bands
        based on cosine similarity thresholds:

        - **Near field** (similarity >= ``near_threshold``): Direct semantic
          associates -- near-synonyms, morphological variants, tightly
          co-occurring concepts.
        - **Mid field** (``mid_threshold`` <= similarity < ``near_threshold``):
          Broader thematic associates -- words from the same theological
          domain or discourse context.
        - **Far field** (similarity < ``mid_threshold``): Peripheral
          associates -- loosely related terms at the boundary of the
          semantic field.

        Parameters
        ----------
        word : str
            Center word for the semantic field.
        near_threshold : float
            Minimum cosine similarity for the near field.
        mid_threshold : float
            Minimum cosine similarity for the mid field.
        max_neighbors : int
            Maximum total neighbors to include across all bands.

        Returns
        -------
        SemanticFieldMap
            Structured field map with stability annotations.
        """
        word = word.lower()
        if word not in self.embeddings.word_to_idx:
            raise ValueError(f"Word '{word}' not found in embedding vocabulary")

        # Get all similarities
        all_similar = self.embeddings.most_similar(word, top_n=max_neighbors)

        near_field: list[tuple[str, float]] = []
        mid_field: list[tuple[str, float]] = []
        far_field: list[tuple[str, float]] = []

        for other_word, sim in all_similar:
            if sim >= near_threshold:
                near_field.append((other_word, sim))
            elif sim >= mid_threshold:
                mid_field.append((other_word, sim))
            else:
                far_field.append((other_word, sim))

        # Gather stability scores for all neighbors
        stability_scores: dict[str, float] = {}
        for other_word, _ in all_similar:
            pair_key = tuple(sorted([word, other_word]))
            stability = self.embeddings.pair_stability.get(pair_key, 0.0)
            stability_scores[other_word] = stability

        logger.info(
            f"Semantic field for '{word}': "
            f"{len(near_field)} near, {len(mid_field)} mid, {len(far_field)} far"
        )

        return SemanticFieldMap(
            target_word=word,
            near_field=near_field,
            mid_field=mid_field,
            far_field=far_field,
            stability_scores=stability_scores,
        )

    # ------------------------------------------------------------------
    # Semantic distance computation
    # ------------------------------------------------------------------

    def compute_semantic_distances(
        self,
        terms: Optional[list[str]] = None,
        include_stability: bool = True,
    ) -> SemanticDistanceMatrix:
        """
        Compute a pairwise distance matrix between theological terms.

        Produces a symmetric matrix of cosine distances between every pair
        of specified terms. Optionally annotates each cell with the
        bootstrap stability of that word pair's relationship.

        Parameters
        ----------
        terms : list[str], optional
            Terms to include in the matrix. If None, uses all contested
            and extended theological terms that are present in the
            embedding vocabulary.
        include_stability : bool
            Whether to include pair stability annotations.

        Returns
        -------
        SemanticDistanceMatrix
            Distance and similarity matrices with term labels.
        """
        if terms is None:
            # Collect all theological terms present in vocabulary
            terms = []
            for english, greek in self.theological_terms:
                if english in self.embeddings.word_to_idx:
                    terms.append(english)
                if greek in self.embeddings.word_to_idx:
                    terms.append(greek)
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique_terms: list[str] = []
            for t in terms:
                if t not in seen:
                    unique_terms.append(t)
                    seen.add(t)
            terms = unique_terms

        if len(terms) < 2:
            raise ValueError(
                f"Need at least 2 terms in vocabulary for distance matrix, "
                f"found {len(terms)}: {terms}"
            )

        n = len(terms)
        distances = np.zeros((n, n))
        similarities = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.embeddings.similarity(terms[i], terms[j])
                dist = 1.0 - sim if not np.isnan(sim) else float("nan")
                distances[i, j] = dist
                distances[j, i] = dist
                similarities[i, j] = sim
                similarities[j, i] = sim

        # Diagonal: zero distance, perfect similarity
        np.fill_diagonal(similarities, 1.0)

        # Build stability matrix if requested
        stability_matrix: Optional[NDArray] = None
        if include_stability:
            stability_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    pair_key = tuple(sorted([terms[i], terms[j]]))
                    stab = self.embeddings.pair_stability.get(pair_key, 0.0)
                    stability_matrix[i, j] = stab
                    stability_matrix[j, i] = stab
            np.fill_diagonal(stability_matrix, 1.0)

        logger.info(
            f"Computed {n}x{n} distance matrix for terms: {terms}"
        )

        return SemanticDistanceMatrix(
            terms=terms,
            distances=distances,
            similarities=similarities,
            stability=stability_matrix,
        )

    # ------------------------------------------------------------------
    # Bridging term discovery
    # ------------------------------------------------------------------

    def find_bridging_terms(
        self,
        cluster_a_words: list[str],
        cluster_b_words: list[str],
        top_n: int = 10,
        min_similarity: float = 0.2,
    ) -> list[tuple[str, float, float, float]]:
        """
        Find words that bridge two semantic clusters.

        A bridging term is one that has moderate similarity to both clusters
        rather than high similarity to just one. These words may represent
        conceptual links in Paul's theology -- terms that mediate between
        two distinct semantic domains.

        For example, if "faith" clusters with trust-related terms and
        "law" clusters with Torah-related terms, a bridging term might be
        "righteousness" if Paul uses it in contexts relating to both.

        Parameters
        ----------
        cluster_a_words : list[str]
            Words in the first semantic cluster.
        cluster_b_words : list[str]
            Words in the second semantic cluster.
        top_n : int
            Maximum number of bridging terms to return.
        min_similarity : float
            Minimum average similarity to both clusters for a word to
            qualify as a bridge.

        Returns
        -------
        list[tuple[str, float, float, float]]
            List of (word, sim_to_a, sim_to_b, bridge_score) tuples,
            sorted by bridge_score descending. The bridge score is
            ``min(sim_to_a, sim_to_b)`` -- high values indicate balanced
            proximity to both clusters.
        """
        # Filter to words actually in vocabulary
        cluster_a = [w for w in cluster_a_words if w in self.embeddings.word_to_idx]
        cluster_b = [w for w in cluster_b_words if w in self.embeddings.word_to_idx]

        if not cluster_a or not cluster_b:
            logger.warning("One or both clusters have no words in vocabulary")
            return []

        # Compute centroids for each cluster
        a_indices = [self.embeddings.word_to_idx[w] for w in cluster_a]
        b_indices = [self.embeddings.word_to_idx[w] for w in cluster_b]
        centroid_a = np.mean(self.embeddings.mean_embeddings[a_indices], axis=0)
        centroid_b = np.mean(self.embeddings.mean_embeddings[b_indices], axis=0)

        # Exclude cluster members from candidate bridges
        excluded = set(cluster_a + cluster_b)

        bridges: list[tuple[str, float, float, float]] = []
        for word in self.embeddings.vocabulary:
            if word in excluded:
                continue

            idx = self.embeddings.word_to_idx[word]
            vec = self.embeddings.mean_embeddings[idx]

            sim_a = 1.0 - cosine(vec, centroid_a)
            sim_b = 1.0 - cosine(vec, centroid_b)

            # Bridge score: minimum similarity to either cluster
            # High bridge score = balanced proximity to both
            bridge_score = min(sim_a, sim_b)

            if bridge_score >= min_similarity:
                bridges.append((word, float(sim_a), float(sim_b), float(bridge_score)))

        bridges.sort(key=lambda x: -x[3])
        return bridges[:top_n]

    # ------------------------------------------------------------------
    # Contested term analysis
    # ------------------------------------------------------------------

    def analyze_contested_terms(
        self,
        top_n_neighbors: int = 15,
    ) -> dict[str, dict]:
        """
        Comprehensive analysis of all contested theological terms.

        For each contested term (justification, faith, law, grace), produces
        a summary including nearest neighbors, stability metrics, semantic
        field map, and (if available) cross-epistle variation.

        Parameters
        ----------
        top_n_neighbors : int
            Number of nearest neighbors to report per term.

        Returns
        -------
        dict[str, dict]
            Maps each contested term to a dictionary containing:

            - ``"neighbors"``: list of (word, similarity) tuples
            - ``"stable_neighbors"``: list of (word, similarity, stability)
            - ``"stability_score"``: neighbor stability for this word
            - ``"field_map"``: SemanticFieldMap instance
            - ``"greek_form"``: corresponding Greek transliteration
            - ``"greek_similarity"``: similarity between English and Greek
              forms (if both are in vocabulary)
        """
        results: dict[str, dict] = {}

        for english, greek in CONTESTED_TERMS:
            term_result: dict = {
                "greek_form": greek,
                "greek_similarity": float("nan"),
            }

            # Analyze English form if present
            if english in self.embeddings.word_to_idx:
                term_result["neighbors"] = self.embeddings.most_similar(
                    english, top_n=top_n_neighbors
                )
                term_result["stable_neighbors"] = self.embeddings.stable_neighbors(
                    english, top_n=top_n_neighbors
                )
                term_result["stability_score"] = self.embeddings.neighbor_stability.get(
                    english, 0.0
                )
                term_result["field_map"] = self.generate_semantic_field_map(english)
            else:
                term_result["neighbors"] = []
                term_result["stable_neighbors"] = []
                term_result["stability_score"] = 0.0
                term_result["field_map"] = None
                logger.info(f"Contested term '{english}' not in embedding vocabulary")

            # Check English-Greek similarity
            if (
                english in self.embeddings.word_to_idx
                and greek in self.embeddings.word_to_idx
            ):
                term_result["greek_similarity"] = self.embeddings.similarity(
                    english, greek
                )

            results[english] = term_result

        return results

    # ------------------------------------------------------------------
    # Summary and reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Generate a summary report of the semantic analysis.

        Returns
        -------
        dict
            Summary containing vocabulary size, tracked term counts,
            contested term availability, and basic distance statistics.
        """
        # Check which contested terms are in vocabulary
        contested_available = {}
        for english, greek in CONTESTED_TERMS:
            contested_available[english] = {
                "english_present": english in self.embeddings.word_to_idx,
                "greek_present": greek in self.embeddings.word_to_idx,
            }

        return {
            "vocabulary_size": len(self.embeddings.vocabulary),
            "embedding_dim": self.embeddings.embedding_dim,
            "n_bootstrap_samples": self.embeddings.n_samples,
            "tracked_theological_terms_found": len(self._tracked_terms),
            "tracked_theological_terms": sorted(self._tracked_terms),
            "contested_terms": contested_available,
            "has_cross_epistle_data": self.cross_epistle is not None,
            "mean_neighbor_stability": float(
                np.mean(list(self.embeddings.neighbor_stability.values()))
            ) if self.embeddings.neighbor_stability else 0.0,
        }
