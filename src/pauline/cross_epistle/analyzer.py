"""
Cross-Epistle Analyzer
======================

Treats each Pauline epistle as an independent "sample" from Paul's
theological mind and measures semantic consistency across samples.

Theoretical Foundation (CLT Analogy):
    Each epistle is a sample from Paul's underlying semantic distribution.
    By training embeddings on different subsets of epistles and comparing
    results, we can identify:

    1. STABLE relationships: word pairs that maintain consistent similarity
       regardless of which epistles are included → "true" Pauline semantics

    2. VARIABLE relationships: word pairs whose similarity changes depending
       on which epistles are present → context-dependent usage

    3. EPISTLE-SPECIFIC relationships: word pairs strongly influenced by
       one particular epistle → situational theology

Methods:
    - Leave-one-out cross-validation at epistle level
    - All-subsets analysis (for 7 epistles: 127 non-empty subsets)
    - Stability scoring across epistle combinations
    - Identification of epistles that most influence specific word relationships
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine

from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes

from ..corpus.loader import PaulineCorpus, Epistle
from ..embeddings.trainer import EmbeddingTrainer

logger = logging.getLogger(__name__)


@dataclass
class WordRelationship:
    """A tracked relationship between two words across epistle subsets."""
    word1: str
    word2: str
    similarities: dict[str, float] = field(default_factory=dict)
    # Maps subset_label -> similarity score

    @property
    def mean_similarity(self) -> float:
        if not self.similarities:
            return 0.0
        return float(np.mean(list(self.similarities.values())))

    @property
    def std_similarity(self) -> float:
        if len(self.similarities) < 2:
            return 0.0
        return float(np.std(list(self.similarities.values())))

    @property
    def stability(self) -> float:
        """1 - coefficient of variation. Higher = more stable."""
        mean = self.mean_similarity
        if abs(mean) < 1e-10:
            return 0.0
        return max(0.0, 1.0 - (self.std_similarity / abs(mean)))

    @property
    def range(self) -> float:
        vals = list(self.similarities.values())
        return max(vals) - min(vals) if vals else 0.0

    def most_influential_epistle(self, direction: str = "increase") -> Optional[str]:
        """
        Find which epistle's inclusion most changes this relationship.

        By comparing leave-one-out results, identifies the epistle
        whose removal causes the largest change in similarity.
        """
        if len(self.similarities) < 2:
            return None

        full_sim = self.similarities.get("all", self.mean_similarity)
        max_change = 0.0
        most_influential = None

        for label, sim in self.similarities.items():
            if label.startswith("without_"):
                change = full_sim - sim
                if direction == "increase" and change > max_change:
                    max_change = change
                    most_influential = label.replace("without_", "")
                elif direction == "decrease" and change < -max_change:
                    max_change = abs(change)
                    most_influential = label.replace("without_", "")

        return most_influential


@dataclass
class CrossEpistleResult:
    """Results from cross-epistle analysis."""
    relationships: dict[tuple[str, str], WordRelationship]
    epistle_subsets_analyzed: int
    vocabulary: list[str]

    # Per-epistle influence scores
    epistle_influence: dict[str, dict[str, float]] = field(default_factory=dict)
    # epistle_name -> {word_pair_key -> influence_score}

    def stable_relationships(self, min_stability: float = 0.8) -> list[WordRelationship]:
        """Get relationships that are stable across epistle subsets."""
        return sorted(
            [r for r in self.relationships.values() if r.stability >= min_stability],
            key=lambda r: -r.stability,
        )

    def variable_relationships(self, max_stability: float = 0.5) -> list[WordRelationship]:
        """Get relationships that vary significantly across epistle subsets."""
        return sorted(
            [r for r in self.relationships.values() if r.stability <= max_stability],
            key=lambda r: r.stability,
        )

    def epistle_specific_words(self, epistle: str, min_influence: float = 0.3) -> list[tuple[str, str, float]]:
        """
        Find word pairs most influenced by a specific epistle.

        Returns:
            List of (word1, word2, influence_score) tuples.
        """
        if epistle not in self.epistle_influence:
            return []

        results = []
        for pair_key, score in self.epistle_influence[epistle].items():
            if score >= min_influence:
                words = pair_key.split(":")
                if len(words) == 2:
                    results.append((words[0], words[1], score))

        return sorted(results, key=lambda x: -x[2])


class CrossEpistleAnalyzer:
    """
    Analyzes semantic relationships across different epistle subsets.

    Core Method: Leave-One-Out Cross-Validation
        For each epistle, train embeddings on all OTHER epistles,
        then compare word relationships to the full-corpus baseline.
        This reveals which relationships depend on specific epistles.

    Advanced Method: All-Subsets Analysis
        For small numbers of epistles (7 undisputed), we can analyze
        ALL possible non-empty subsets (2^7 - 1 = 127 combinations).
        This gives the most complete picture of semantic stability.
    """

    def __init__(
        self,
        corpus: PaulineCorpus,
        embedding_dim: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 50,
    ):
        self.corpus = corpus
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.epochs = epochs

    def leave_one_out(
        self,
        target_words: list[str],
    ) -> CrossEpistleResult:
        """
        Leave-one-out cross-validation at epistle level.

        For each epistle E_i:
            1. Train embeddings on corpus WITHOUT E_i
            2. Measure similarity between all target word pairs
            3. Compare to full-corpus baseline

        Args:
            target_words: Words to track relationships between.

        Returns:
            CrossEpistleResult with relationship stability data.
        """
        target_words = [w.lower() for w in target_words]
        relationships: dict[tuple[str, str], WordRelationship] = {}

        # Initialize relationship tracking
        for i, w1 in enumerate(target_words):
            for w2 in target_words[i + 1:]:
                key = (w1, w2)
                relationships[key] = WordRelationship(word1=w1, word2=w2)

        # Train on full corpus first (baseline)
        logger.info("Training baseline embeddings on full corpus...")
        full_model = self._train_model(
            [ep.text for ep in self.corpus.epistles]
        )

        if full_model is None:
            raise ValueError("Failed to train baseline model")

        # Record baseline similarities
        for key in relationships:
            w1, w2 = key
            if w1 in full_model.wv and w2 in full_model.wv:
                sim = full_model.wv.similarity(w1, w2)
                relationships[key].similarities["all"] = float(sim)

        # Leave-one-out iterations
        epistle_influence: dict[str, dict[str, float]] = {}

        for ep in self.corpus.epistles:
            logger.info(f"Training without {ep.name}...")

            remaining_texts = [
                other.text for other in self.corpus.epistles
                if other.name != ep.name
            ]

            model = self._train_model(remaining_texts)
            if model is None:
                logger.warning(f"Failed to train model without {ep.name}")
                continue

            epistle_influence[ep.name] = {}

            for key in relationships:
                w1, w2 = key
                if w1 in model.wv and w2 in model.wv:
                    sim = model.wv.similarity(w1, w2)
                    relationships[key].similarities[f"without_{ep.name}"] = float(sim)

                    # Compute influence: how much does removing this epistle
                    # change the similarity?
                    baseline = relationships[key].similarities.get("all", 0.0)
                    influence = abs(baseline - sim)
                    pair_key = f"{w1}:{w2}"
                    epistle_influence[ep.name][pair_key] = influence

        # Identify common vocabulary across all models
        common_vocab = list(set(target_words) & set(full_model.wv.key_to_index.keys()))

        return CrossEpistleResult(
            relationships=relationships,
            epistle_subsets_analyzed=len(self.corpus.epistles) + 1,
            vocabulary=common_vocab,
            epistle_influence=epistle_influence,
        )

    def all_subsets(
        self,
        target_words: list[str],
        min_subset_size: int = 3,
    ) -> CrossEpistleResult:
        """
        Analyze all possible epistle subsets (for small corpus).

        For 7 undisputed epistles with min_subset_size=3, this trains
        embeddings on C(7,3) + C(7,4) + ... + C(7,7) = 99 subsets.

        This is computationally expensive but gives the most complete
        picture of which relationships are truly stable in Paul.

        Args:
            target_words: Words to track.
            min_subset_size: Minimum number of epistles in a subset.

        Returns:
            CrossEpistleResult with comprehensive stability data.
        """
        target_words = [w.lower() for w in target_words]
        epistles = self.corpus.epistles
        n_epistles = len(epistles)

        relationships: dict[tuple[str, str], WordRelationship] = {}
        for i, w1 in enumerate(target_words):
            for w2 in target_words[i + 1:]:
                relationships[(w1, w2)] = WordRelationship(word1=w1, word2=w2)

        # Generate all subsets of size >= min_subset_size
        total_subsets = 0
        for size in range(min_subset_size, n_epistles + 1):
            total_subsets += len(list(combinations(range(n_epistles), size)))

        logger.info(f"Analyzing {total_subsets} epistle subsets...")

        subset_count = 0
        for size in range(min_subset_size, n_epistles + 1):
            for indices in combinations(range(n_epistles), size):
                subset_epistles = [epistles[i] for i in indices]
                label = "+".join(ep.abbreviation for ep in subset_epistles)
                texts = [ep.text for ep in subset_epistles]

                model = self._train_model(texts)
                if model is None:
                    continue

                for key in relationships:
                    w1, w2 = key
                    if w1 in model.wv and w2 in model.wv:
                        sim = model.wv.similarity(w1, w2)
                        relationships[key].similarities[label] = float(sim)

                subset_count += 1
                if subset_count % 10 == 0:
                    logger.info(f"  Analyzed {subset_count}/{total_subsets} subsets")

        common_vocab = list(set(target_words) & self.corpus.vocabulary)

        return CrossEpistleResult(
            relationships=relationships,
            epistle_subsets_analyzed=subset_count,
            vocabulary=common_vocab,
        )

    def _train_model(self, texts: list[str]) -> Optional[Word2Vec]:
        """Train a Word2Vec model on the given texts."""
        from ..corpus.loader import sent_tokenize, word_tokenize

        sentences = []
        for text in texts:
            for sent in sent_tokenize(text):
                tokens = word_tokenize(sent)
                if len(tokens) >= 3:
                    sentences.append(tokens)

        if len(sentences) < 5:
            return None

        try:
            model = Word2Vec(
                sentences=sentences,
                vector_size=self.embedding_dim,
                window=self.window,
                min_count=self.min_count,
                sg=1,
                epochs=self.epochs,
                workers=4,
                seed=42,
            )
            return model
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None
