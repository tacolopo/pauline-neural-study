"""
Permutation Language Model (PLM)
================================

Trains a language model to predict the original sentence ordering within
Pauline pericopes, using all valid permutations of sentences as training
data.

Theoretical Foundation:
-----------------------
Paul's pericopes (coherent theological argument units) have an internal
logic: premise -> development -> conclusion. If we scramble the sentences
within a pericope, a well-trained model should be able to recover the
original order — because Paul's argumentation follows consistent
rhetorical patterns.

This module exploits that structure to generate factorial growth in
training data while using **only** Paul's actual words:

    - A pericope with N sentences generates N! permutations.
    - Each permutation is a valid training example: the model learns to
      assign high probability to the original ordering and lower
      probability to scrambled orderings.
    - For a typical pericope of 5 sentences: 5! = 120 training samples
      from a single passage.
    - Across all ~50 pericopes: tens of thousands of training samples,
      all containing exclusively Pauline vocabulary.

The training objective is inspired by XLNet's permutation language
modeling (Yang et al., 2019), but applied at the sentence level rather
than the token level, and within a closed corpus rather than a general
pre-training setup.

Architecture:
    - Sentence encoder: maps each sentence to a fixed-size embedding
      (using TF-IDF + SVD or a small transformer).
    - Positional scorer: for each sentence embedding, predicts the
      probability of each position (1..N) in the original ordering.
    - Training: cross-entropy loss between predicted position
      distributions and true positions.
    - The model learns Paul's rhetorical patterns: what kind of
      sentence comes first (thesis), what develops the argument,
      and what concludes it.

Reference:
    - Yang, Z. et al. "XLNet: Generalized Autoregressive Pretraining
      for Language Understanding" (2019)
    - Barzilay, R. & Lapata, M. "Modeling Local Coherence: An
      Entity-Based Approach" (2008) — sentence ordering as a probe
      for discourse coherence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from ..corpus.loader import PaulineCorpus, Pericope, word_tokenize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PermutationSample:
    """A single training sample: a permuted pericope.

    Attributes:
        pericope_label: Identifier for the source pericope.
        sentences: The sentences in permuted order.
        permutation: The permutation applied (tuple of original indices).
        is_original: True if this is the canonical (original) ordering.
        target_positions: The correct position for each sentence in the
            permuted sequence (i.e., the inverse permutation).
    """
    pericope_label: str
    sentences: list[str]
    permutation: tuple[int, ...]
    is_original: bool
    target_positions: list[int]


@dataclass
class PLMDataset:
    """Complete dataset of permutation samples for training.

    Attributes:
        samples: All permutation samples.
        n_pericopes: Number of source pericopes.
        n_original: Number of original-order samples.
        n_permuted: Number of permuted samples.
        max_sentences: Maximum number of sentences in any pericope.
        factorial_expansion: Ratio of total samples to original pericopes.
    """
    samples: list[PermutationSample] = field(default_factory=list)
    n_pericopes: int = 0
    n_original: int = 0
    n_permuted: int = 0
    max_sentences: int = 0
    factorial_expansion: float = 0.0


@dataclass
class SentenceEmbedding:
    """TF-IDF + SVD embedding for a sentence.

    Attributes:
        text: Original sentence text.
        vector: Dense embedding vector.
        pericope_label: Source pericope.
        position: Original position within the pericope.
    """
    text: str
    vector: NDArray
    pericope_label: str
    position: int


@dataclass
class PLMResult:
    """Results from training and evaluating the permutation LM.

    Attributes:
        train_accuracy: Fraction of training samples where the model
            correctly predicted the original ordering.
        eval_accuracy: Fraction of held-out pericopes correctly ordered.
        kendall_tau_mean: Mean Kendall tau correlation between predicted
            and true orderings (1.0 = perfect, -1.0 = reversed).
        position_accuracy: Per-position accuracy (how often the model
            correctly places a sentence at position k).
        weights: Learned weight matrix of the positional scorer.
        dataset_stats: Summary of the training dataset.
        pericope_scores: Per-pericope Kendall tau values.
    """
    train_accuracy: float
    eval_accuracy: float
    kendall_tau_mean: float
    position_accuracy: dict[int, float]
    weights: Optional[NDArray] = None
    dataset_stats: Optional[dict] = None
    pericope_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main PLM class
# ---------------------------------------------------------------------------

class PermutationLM:
    """
    Permutation language model for Pauline pericopes.

    Generates all (or a sampled subset of) sentence permutations for
    each pericope, encodes sentences via TF-IDF + SVD, and trains a
    linear positional scorer to predict original sentence ordering.

    Usage::

        from pauline.corpus.loader import PaulineCorpus
        from pauline.permutation.plm import PermutationLM

        corpus = PaulineCorpus.from_json("pauline_corpus.json")
        plm = PermutationLM(corpus)
        dataset = plm.build_dataset(max_perm_per_pericope=100)
        result = plm.train(dataset)

        print(f"Kendall tau: {result.kendall_tau_mean:.3f}")
        print(f"Accuracy: {result.eval_accuracy:.1%}")
    """

    def __init__(
        self,
        corpus: PaulineCorpus,
        embedding_dim: int = 64,
        seed: Optional[int] = None,
    ):
        self.corpus = corpus
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Will be populated during prepare()
        self._pericopes: list[Pericope] = []
        self._tfidf_matrix: Optional[csr_matrix] = None
        self._svd_components: Optional[NDArray] = None
        self._vocab_list: list[str] = []
        self._idf: Optional[NDArray] = None
        self._prepared = False

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        max_perm_per_pericope: int = 120,
        min_sentences: int = 3,
        max_sentences: int = 15,
        include_original: bool = True,
    ) -> PLMDataset:
        """
        Build the permutation training dataset from the corpus.

        For each pericope with N sentences (min_sentences <= N <= max_sentences):
            - If N! <= max_perm_per_pericope: use ALL permutations.
            - Otherwise: sample max_perm_per_pericope random permutations
              (always including the original).

        Args:
            max_perm_per_pericope: Maximum number of permutations per
                pericope. Controls memory usage.
            min_sentences: Skip pericopes with fewer sentences.
            max_sentences: Skip pericopes with more sentences (factorial
                explosion control).
            include_original: Always include the original ordering.

        Returns:
            PLMDataset ready for training.
        """
        self._collect_pericopes()

        dataset = PLMDataset()
        max_sent = 0

        for pericope in self._pericopes:
            sents = pericope.sentences
            n = len(sents)

            if n < min_sentences or n > max_sentences:
                continue

            max_sent = max(max_sent, n)
            dataset.n_pericopes += 1

            n_factorial = math.factorial(n)
            use_all = n_factorial <= max_perm_per_pericope

            if use_all:
                # Generate all permutations
                perms = list(permutations(range(n)))
            else:
                # Sample random permutations
                perms_set: set[tuple[int, ...]] = set()
                # Always include the identity
                if include_original:
                    perms_set.add(tuple(range(n)))

                while len(perms_set) < max_perm_per_pericope:
                    perm = tuple(self.rng.permutation(n).tolist())
                    perms_set.add(perm)

                perms = list(perms_set)

            label = f"{pericope.book}: {pericope.label}"

            for perm in perms:
                is_orig = perm == tuple(range(n))
                permuted_sents = [sents[i] for i in perm]

                # Target positions: for each sentence in the permuted
                # order, what was its original position?
                # This is the inverse permutation.
                inv_perm = [0] * n
                for new_pos, old_pos in enumerate(perm):
                    inv_perm[new_pos] = old_pos

                sample = PermutationSample(
                    pericope_label=label,
                    sentences=permuted_sents,
                    permutation=perm,
                    is_original=is_orig,
                    target_positions=inv_perm,
                )
                dataset.samples.append(sample)

                if is_orig:
                    dataset.n_original += 1
                else:
                    dataset.n_permuted += 1

        dataset.max_sentences = max_sent
        if dataset.n_pericopes > 0:
            dataset.factorial_expansion = (
                len(dataset.samples) / dataset.n_pericopes
            )

        logger.info(
            f"Built PLM dataset: {dataset.n_pericopes} pericopes, "
            f"{len(dataset.samples)} samples "
            f"({dataset.factorial_expansion:.1f}x expansion), "
            f"max {dataset.max_sentences} sentences"
        )

        return dataset

    # ------------------------------------------------------------------
    # Sentence embedding (TF-IDF + SVD)
    # ------------------------------------------------------------------

    def prepare_embeddings(self) -> None:
        """
        Build TF-IDF vocabulary and SVD projection from the corpus.

        This creates a sentence embedding pipeline that maps any
        sentence (composed of Pauline vocabulary) to a dense vector
        without requiring external pre-trained models.
        """
        self._collect_pericopes()

        # Gather all unique sentences across all pericopes
        all_sentences: list[str] = []
        for pericope in self._pericopes:
            all_sentences.extend(pericope.sentences)

        if not all_sentences:
            # Fall back to corpus-level sentences
            all_sentences = [s for _, s in self.corpus.all_sentences]

        logger.info(f"Building TF-IDF from {len(all_sentences)} sentences")

        # Build vocabulary
        doc_freq: dict[str, int] = {}
        tokenized: list[list[str]] = []

        for sent in all_sentences:
            tokens = word_tokenize(sent)
            tokens = [t for t in tokens if t.isalpha()]
            tokenized.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        # Filter to words appearing in at least 2 documents
        n_docs = len(all_sentences)
        self._vocab_list = sorted(
            w for w, df in doc_freq.items() if df >= 2
        )
        vocab_idx = {w: i for i, w in enumerate(self._vocab_list)}
        v = len(self._vocab_list)

        if v == 0:
            raise ValueError("No vocabulary items survived filtering")

        # Compute IDF
        self._idf = np.log(n_docs / (1 + np.array([
            doc_freq.get(w, 1) for w in self._vocab_list
        ], dtype=np.float64)))

        # Build TF-IDF matrix (n_docs x vocab_size)
        rows, cols, data = [], [], []
        for doc_i, tokens in enumerate(tokenized):
            tf: dict[int, int] = {}
            for t in tokens:
                if t in vocab_idx:
                    idx = vocab_idx[t]
                    tf[idx] = tf.get(idx, 0) + 1
            for idx, count in tf.items():
                rows.append(doc_i)
                cols.append(idx)
                data.append(count * self._idf[idx])

        self._tfidf_matrix = csr_matrix(
            (data, (rows, cols)), shape=(n_docs, v)
        )

        # SVD dimensionality reduction
        k = min(self.embedding_dim, v - 1, n_docs - 1)
        if k < 1:
            k = 1

        u, s, vt = svds(self._tfidf_matrix.astype(np.float64), k=k)
        # Components for projection: project new docs via (tfidf @ V) / s
        self._svd_components = vt.T  # shape: (vocab_size, k)
        self._svd_singular = s  # shape: (k,)

        self._prepared = True
        logger.info(
            f"TF-IDF + SVD ready: {v} vocab, "
            f"{k}-dim embeddings"
        )

    def embed_sentence(self, sentence: str) -> NDArray:
        """
        Embed a single sentence to a dense vector via TF-IDF + SVD.

        Args:
            sentence: Input sentence text.

        Returns:
            1-D array of shape (embedding_dim,).
        """
        if not self._prepared:
            self.prepare_embeddings()

        vocab_idx = {w: i for i, w in enumerate(self._vocab_list)}
        tokens = word_tokenize(sentence)
        tokens = [t for t in tokens if t.isalpha() and t in vocab_idx]

        v = len(self._vocab_list)
        tfidf_vec = np.zeros(v)
        for t in tokens:
            idx = vocab_idx[t]
            tfidf_vec[idx] += self._idf[idx]

        # Project through SVD
        embedded = tfidf_vec @ self._svd_components
        # Normalize by singular values for stability
        if self._svd_singular is not None:
            embedded = embedded / (self._svd_singular + 1e-10)

        # L2 normalize
        norm = np.linalg.norm(embedded)
        if norm > 1e-10:
            embedded /= norm

        return embedded

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: PLMDataset,
        learning_rate: float = 0.01,
        n_epochs: int = 50,
        eval_fraction: float = 0.2,
    ) -> PLMResult:
        """
        Train the permutation language model.

        The model is a linear positional scorer:
            score(sentence_i, position_k) = embedding_i @ W[:, k]

        For each permuted pericope, the model predicts a position
        distribution for each sentence via softmax over scores.
        Training minimizes cross-entropy against the true positions.

        Args:
            dataset: PLMDataset from ``build_dataset()``.
            learning_rate: Learning rate for gradient descent.
            n_epochs: Number of training epochs.
            eval_fraction: Fraction of pericopes held out for evaluation.

        Returns:
            PLMResult with accuracy, Kendall tau, and other metrics.
        """
        if not self._prepared:
            self.prepare_embeddings()

        if not dataset.samples:
            raise ValueError("Dataset is empty")

        max_n = dataset.max_sentences
        dim = self._svd_components.shape[1] if self._svd_components is not None else self.embedding_dim

        # Split by pericope for train/eval
        pericope_labels = sorted(set(s.pericope_label for s in dataset.samples))
        n_eval = max(1, int(len(pericope_labels) * eval_fraction))
        self.rng.shuffle(pericope_labels)
        eval_pericopes = set(pericope_labels[:n_eval])
        train_pericopes = set(pericope_labels[n_eval:])

        train_samples = [
            s for s in dataset.samples if s.pericope_label in train_pericopes
        ]
        eval_samples = [
            s for s in dataset.samples if s.pericope_label in eval_pericopes
        ]

        logger.info(
            f"Training: {len(train_samples)} samples from "
            f"{len(train_pericopes)} pericopes; "
            f"Eval: {len(eval_samples)} samples from "
            f"{len(eval_pericopes)} pericopes"
        )

        # Initialize weight matrix: (embedding_dim, max_positions)
        W = self.rng.normal(0, 0.01, size=(dim, max_n)).astype(np.float64)

        # --- Training loop ---
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_correct = 0
            n_total = 0

            # Shuffle training samples
            indices = self.rng.permutation(len(train_samples))

            for idx in indices:
                sample = train_samples[idx]
                n_sent = len(sample.sentences)
                targets = np.array(sample.target_positions)

                # Embed all sentences
                embeddings = np.array([
                    self.embed_sentence(s) for s in sample.sentences
                ])  # (n_sent, dim)

                # Compute scores: (n_sent, max_n)
                logits = embeddings @ W[:, :n_sent]  # (n_sent, n_sent)

                # Softmax per sentence
                logits_shifted = logits - logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits_shifted)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                # Cross-entropy loss
                for s_idx in range(n_sent):
                    target_pos = targets[s_idx]
                    if target_pos < n_sent:
                        epoch_loss -= np.log(probs[s_idx, target_pos] + 1e-10)

                # Predicted ordering
                predicted = np.argmax(logits, axis=1)
                if np.array_equal(predicted, targets):
                    n_correct += 1
                n_total += 1

                # Gradient update (softmax cross-entropy gradient)
                grad_logits = probs.copy()  # (n_sent, n_sent)
                for s_idx in range(n_sent):
                    target_pos = targets[s_idx]
                    if target_pos < n_sent:
                        grad_logits[s_idx, target_pos] -= 1.0

                # dL/dW for the positions used
                grad_W = np.zeros_like(W)
                grad_W[:, :n_sent] += embeddings.T @ grad_logits

                W -= learning_rate * grad_W

            train_acc = n_correct / max(n_total, 1)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"loss={epoch_loss:.2f}, "
                    f"train_acc={train_acc:.3f}"
                )

        # --- Evaluation ---
        eval_acc, kendall_mean, position_acc, pericope_scores = self._evaluate(
            eval_samples, W, max_n
        )

        stats = {
            "n_pericopes": dataset.n_pericopes,
            "n_train_samples": len(train_samples),
            "n_eval_samples": len(eval_samples),
            "factorial_expansion": dataset.factorial_expansion,
            "max_sentences": max_n,
        }

        return PLMResult(
            train_accuracy=train_acc,
            eval_accuracy=eval_acc,
            kendall_tau_mean=kendall_mean,
            position_accuracy=position_acc,
            weights=W,
            dataset_stats=stats,
            pericope_scores=pericope_scores,
        )

    # ------------------------------------------------------------------
    # Prediction and evaluation
    # ------------------------------------------------------------------

    def predict_ordering(
        self,
        sentences: list[str],
        weights: NDArray,
    ) -> list[int]:
        """
        Predict the original position for each sentence.

        Uses the Hungarian algorithm to find the optimal assignment
        (one sentence per position, maximizing total score).

        Args:
            sentences: List of sentences in arbitrary order.
            weights: Trained weight matrix.

        Returns:
            List of predicted original positions for each sentence.
        """
        from scipy.optimize import linear_sum_assignment

        if not self._prepared:
            self.prepare_embeddings()

        n = len(sentences)
        embeddings = np.array([self.embed_sentence(s) for s in sentences])
        logits = embeddings @ weights[:, :n]

        # Use Hungarian algorithm for optimal assignment
        # (maximize score = minimize negative score)
        row_ind, col_ind = linear_sum_assignment(-logits)

        predicted = [0] * n
        for r, c in zip(row_ind, col_ind):
            predicted[r] = c

        return predicted

    def _evaluate(
        self,
        samples: list[PermutationSample],
        weights: NDArray,
        max_n: int,
    ) -> tuple[float, float, dict[int, float], dict[str, float]]:
        """
        Evaluate the model on held-out samples.

        Returns:
            Tuple of (exact_accuracy, mean_kendall_tau,
            per_position_accuracy, per_pericope_scores).
        """
        from scipy.stats import kendalltau

        if not samples:
            return 0.0, 0.0, {}, {}

        # Group by pericope and evaluate only original orderings
        original_samples = [s for s in samples if s.is_original]
        if not original_samples:
            # Use all samples if no originals flagged
            original_samples = samples

        n_correct = 0
        taus = []
        position_correct: dict[int, int] = {}
        position_total: dict[int, int] = {}
        pericope_scores: dict[str, float] = {}

        for sample in original_samples:
            predicted = self.predict_ordering(sample.sentences, weights)
            targets = sample.target_positions

            # Exact match
            if predicted == targets:
                n_correct += 1

            # Kendall tau
            tau, _ = kendalltau(predicted, targets)
            if not np.isnan(tau):
                taus.append(tau)
                pericope_scores[sample.pericope_label] = float(tau)

            # Per-position accuracy
            for pos in range(len(targets)):
                position_total[pos] = position_total.get(pos, 0) + 1
                if pos < len(predicted) and predicted[pos] == targets[pos]:
                    position_correct[pos] = position_correct.get(pos, 0) + 1

        exact_acc = n_correct / max(len(original_samples), 1)
        mean_tau = float(np.mean(taus)) if taus else 0.0
        pos_acc = {
            pos: position_correct.get(pos, 0) / total
            for pos, total in position_total.items()
        }

        logger.info(
            f"Evaluation: exact_acc={exact_acc:.3f}, "
            f"kendall_tau={mean_tau:.3f}, "
            f"{len(original_samples)} pericopes evaluated"
        )

        return exact_acc, mean_tau, pos_acc, pericope_scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_pericopes(self) -> None:
        """Collect all pericopes from the corpus, falling back to
        chapters as pseudo-pericopes if no pericopes are defined."""
        if self._pericopes:
            return

        for ep in self.corpus.epistles:
            if ep.pericopes:
                self._pericopes.extend(ep.pericopes)
            else:
                # Create pseudo-pericopes from chapters
                for ch_num, verses in ep.chapters.items():
                    pseudo = Pericope(
                        book=ep.name,
                        label=f"Chapter {ch_num}",
                        verses=verses,
                    )
                    self._pericopes.append(pseudo)

        logger.info(f"Collected {len(self._pericopes)} pericopes/chapters")
