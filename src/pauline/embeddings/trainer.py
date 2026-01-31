"""
Embedding Trainer
=================

Trains word embeddings on Pauline corpus bootstrap samples.

The core insight: by training embeddings on many bootstrap samples
and aggregating the results, we can identify stable semantic
relationships in Paul's vocabulary — relationships that persist
regardless of which specific texts are sampled.

This implements the Antoniak & Mimno approach: train Word2Vec on
each bootstrap sample, then measure the stability of nearest-neighbor
relationships across all samples. Words that consistently cluster
together across samples share genuine semantic affinity in Paul's usage.

Architecture:
    - Word2Vec (Skip-gram or CBOW) for individual sample training
    - Procrustes alignment to make embeddings comparable across samples
    - Aggregation via averaging aligned embedding matrices
    - Stability metrics for word pair relationships
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cosine
from scipy.linalg import orthogonal_procrustes

from gensim.models import Word2Vec

from ..bootstrap.sampler import BootstrapResult, BootstrapSample

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Results from training embeddings on bootstrap samples."""
    # Core results
    vocabulary: list[str]
    word_to_idx: dict[str, int]
    mean_embeddings: NDArray  # shape: (vocab_size, embedding_dim)

    # Stability metrics
    neighbor_stability: dict[str, float]  # word -> stability score [0,1]
    pair_stability: dict[tuple[str, str], float]  # (word1, word2) -> stability

    # Per-sample embeddings (optional, memory-intensive)
    sample_embeddings: Optional[list[NDArray]] = None

    # Metadata
    n_samples: int = 0
    embedding_dim: int = 100
    min_count: int = 2

    def similarity(self, word1: str, word2: str) -> float:
        """Cosine similarity between two words in the mean embedding space."""
        if word1 not in self.word_to_idx or word2 not in self.word_to_idx:
            return float("nan")
        idx1 = self.word_to_idx[word1]
        idx2 = self.word_to_idx[word2]
        return 1.0 - cosine(self.mean_embeddings[idx1], self.mean_embeddings[idx2])

    def most_similar(self, word: str, top_n: int = 10) -> list[tuple[str, float]]:
        """Find most similar words in the mean embedding space."""
        if word not in self.word_to_idx:
            return []
        idx = self.word_to_idx[word]
        vec = self.mean_embeddings[idx]

        similarities = []
        for other_word, other_idx in self.word_to_idx.items():
            if other_word == word:
                continue
            sim = 1.0 - cosine(vec, self.mean_embeddings[other_idx])
            similarities.append((other_word, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_n]

    def stable_neighbors(self, word: str, top_n: int = 10, min_stability: float = 0.5) -> list[tuple[str, float, float]]:
        """
        Find words that are both similar AND stable neighbors.

        Returns:
            List of (word, similarity, stability) tuples, sorted by
            combined score (similarity * stability).
        """
        similar = self.most_similar(word, top_n=top_n * 3)
        results = []
        for other_word, sim in similar:
            pair_key = tuple(sorted([word, other_word]))
            stability = self.pair_stability.get(pair_key, 0.0)
            if stability >= min_stability:
                results.append((other_word, sim, stability))

        results.sort(key=lambda x: -(x[1] * x[2]))
        return results[:top_n]

    def save(self, path: str | Path) -> None:
        """Save embedding results to disk."""
        path = Path(path)
        np.savez_compressed(
            path,
            mean_embeddings=self.mean_embeddings,
            vocabulary=np.array(self.vocabulary),
            neighbor_stability=np.array(
                [self.neighbor_stability.get(w, 0.0) for w in self.vocabulary]
            ),
        )
        logger.info(f"Embeddings saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingResult:
        """Load embedding results from disk."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        vocabulary = data["vocabulary"].tolist()
        word_to_idx = {w: i for i, w in enumerate(vocabulary)}
        stability_scores = data["neighbor_stability"]

        return cls(
            vocabulary=vocabulary,
            word_to_idx=word_to_idx,
            mean_embeddings=data["mean_embeddings"],
            neighbor_stability={w: stability_scores[i] for i, w in enumerate(vocabulary)},
            pair_stability={},
            embedding_dim=data["mean_embeddings"].shape[1],
        )


class EmbeddingTrainer:
    """
    Trains word embeddings across bootstrap samples and measures stability.

    For each bootstrap sample:
        1. Tokenize text into sentences/words
        2. Train Word2Vec model
        3. Extract embedding matrix

    After all samples are trained:
        4. Align embeddings using Procrustes analysis
        5. Compute mean embedding across all aligned samples
        6. Measure neighbor stability for each word
        7. Measure pairwise stability for word pairs of interest
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,  # 1 = Skip-gram, 0 = CBOW
        epochs: int = 50,
        workers: int = 4,
        top_n_neighbors: int = 10,
    ):
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.workers = workers
        self.top_n_neighbors = top_n_neighbors

    def train_on_bootstrap(
        self,
        bootstrap_result: BootstrapResult,
        target_words: Optional[list[str]] = None,
        store_all_embeddings: bool = False,
    ) -> EmbeddingResult:
        """
        Train embeddings on all bootstrap samples and compute stability.

        Args:
            bootstrap_result: Bootstrap samples to train on.
            target_words: Words of special interest for pair stability
                         analysis. If None, uses top vocabulary words.
            store_all_embeddings: Whether to keep per-sample embedding
                                matrices (memory-intensive).

        Returns:
            EmbeddingResult with mean embeddings and stability metrics.
        """
        import nltk

        n_samples = len(bootstrap_result.samples)
        logger.info(f"Training embeddings on {n_samples} bootstrap samples...")

        # Phase 1: Train individual models and extract embeddings
        models: list[Word2Vec] = []
        for i, sample in enumerate(bootstrap_result.samples):
            sentences = [
                nltk.word_tokenize(text.lower())
                for text in sample.texts
            ]

            # Filter out very short sentences
            sentences = [s for s in sentences if len(s) >= 3]

            if not sentences:
                logger.warning(f"Sample {i} has no valid sentences, skipping")
                continue

            model = Word2Vec(
                sentences=sentences,
                vector_size=self.embedding_dim,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                epochs=self.epochs,
                workers=self.workers,
                seed=42 + i,
            )
            models.append(model)

            if (i + 1) % 50 == 0:
                logger.info(f"  Trained {i + 1}/{n_samples} models")

        if not models:
            raise ValueError("No valid models were trained")

        logger.info(f"Trained {len(models)} models successfully")

        # Phase 2: Find common vocabulary across all models
        common_vocab = set(models[0].wv.key_to_index.keys())
        for model in models[1:]:
            common_vocab &= set(model.wv.key_to_index.keys())

        # Remove punctuation-only tokens
        common_vocab = {
            w for w in common_vocab
            if any(c.isalpha() for c in w)
        }

        vocabulary = sorted(common_vocab)
        word_to_idx = {w: i for i, w in enumerate(vocabulary)}

        logger.info(f"Common vocabulary size: {len(vocabulary)} words")

        if len(vocabulary) < 10:
            logger.warning(
                "Very small common vocabulary. Consider lowering min_count "
                "or using more bootstrap samples."
            )

        # Phase 3: Extract and align embeddings
        vocab_size = len(vocabulary)
        all_embeddings: list[NDArray] = []

        # Use first model as reference for Procrustes alignment
        ref_matrix = np.array([
            models[0].wv[w] for w in vocabulary
        ])

        for i, model in enumerate(models):
            # Extract embedding matrix for common vocabulary
            matrix = np.array([model.wv[w] for w in vocabulary])

            # Procrustes alignment to reference
            if i == 0:
                aligned = matrix
            else:
                aligned = self._procrustes_align(matrix, ref_matrix)

            all_embeddings.append(aligned)

        # Phase 4: Compute mean embeddings
        embedding_stack = np.stack(all_embeddings)  # (n_samples, vocab_size, dim)
        mean_embeddings = np.mean(embedding_stack, axis=0)  # (vocab_size, dim)

        # Phase 5: Compute stability metrics
        logger.info("Computing stability metrics...")

        neighbor_stability = self._compute_neighbor_stability(
            all_embeddings, vocabulary, word_to_idx
        )

        # Compute pair stability for target words
        if target_words is None:
            target_words = vocabulary[:50]  # Top 50 by frequency
        else:
            target_words = [w for w in target_words if w in word_to_idx]

        pair_stability = self._compute_pair_stability(
            all_embeddings, vocabulary, word_to_idx, target_words
        )

        return EmbeddingResult(
            vocabulary=vocabulary,
            word_to_idx=word_to_idx,
            mean_embeddings=mean_embeddings,
            neighbor_stability=neighbor_stability,
            pair_stability=pair_stability,
            sample_embeddings=all_embeddings if store_all_embeddings else None,
            n_samples=len(models),
            embedding_dim=self.embedding_dim,
            min_count=self.min_count,
        )

    def _procrustes_align(
        self, source: NDArray, target: NDArray
    ) -> NDArray:
        """
        Align source embedding matrix to target using Procrustes analysis.

        Finds the orthogonal rotation matrix R that minimizes
        ||source @ R - target||_F, preserving distances within
        each embedding space while making them comparable.
        """
        R, _ = orthogonal_procrustes(source, target)
        return source @ R

    def _compute_neighbor_stability(
        self,
        all_embeddings: list[NDArray],
        vocabulary: list[str],
        word_to_idx: dict[str, int],
    ) -> dict[str, float]:
        """
        Compute neighbor stability for each word.

        For each word, find its top-k nearest neighbors in each
        bootstrap sample. Stability = average Jaccard similarity
        of neighbor sets across all pairs of samples.

        A stability score of 1.0 means the word always has the
        same neighbors; 0.0 means neighbors are completely random.
        """
        k = self.top_n_neighbors
        n_samples = len(all_embeddings)
        stability: dict[str, float] = {}

        for word in vocabulary:
            idx = word_to_idx[word]
            neighbor_sets: list[set[int]] = []

            for emb in all_embeddings:
                vec = emb[idx]
                # Compute distances to all other words
                distances = np.array([
                    cosine(vec, emb[j]) if j != idx else float("inf")
                    for j in range(len(vocabulary))
                ])
                # Get top-k nearest neighbor indices
                top_k = set(np.argsort(distances)[:k].tolist())
                neighbor_sets.append(top_k)

            # Average pairwise Jaccard similarity
            jaccard_sum = 0.0
            n_pairs = 0
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    intersection = len(neighbor_sets[i] & neighbor_sets[j])
                    union = len(neighbor_sets[i] | neighbor_sets[j])
                    if union > 0:
                        jaccard_sum += intersection / union
                    n_pairs += 1

            stability[word] = jaccard_sum / max(n_pairs, 1)

        return stability

    def _compute_pair_stability(
        self,
        all_embeddings: list[NDArray],
        vocabulary: list[str],
        word_to_idx: dict[str, int],
        target_words: list[str],
    ) -> dict[tuple[str, str], float]:
        """
        Compute stability of similarity between specific word pairs.

        For each pair of target words, compute their cosine similarity
        in each bootstrap sample. Stability = 1 - coefficient of variation
        of the similarity values across samples.

        High stability means the relationship between two words is
        consistent regardless of which texts are sampled.
        """
        pair_stability: dict[tuple[str, str], float] = {}

        for i, word1 in enumerate(target_words):
            if word1 not in word_to_idx:
                continue
            idx1 = word_to_idx[word1]

            for word2 in target_words[i + 1:]:
                if word2 not in word_to_idx:
                    continue
                idx2 = word_to_idx[word2]

                # Compute similarity in each sample
                sims = []
                for emb in all_embeddings:
                    sim = 1.0 - cosine(emb[idx1], emb[idx2])
                    sims.append(sim)

                sims = np.array(sims)
                mean_sim = np.mean(sims)
                std_sim = np.std(sims)

                # Stability: 1 - CV (coefficient of variation)
                if abs(mean_sim) > 1e-10:
                    cv = std_sim / abs(mean_sim)
                    stab = max(0.0, 1.0 - cv)
                else:
                    stab = 0.0

                pair_key = tuple(sorted([word1, word2]))
                pair_stability[pair_key] = stab

        return pair_stability
