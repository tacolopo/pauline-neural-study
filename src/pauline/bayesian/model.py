"""
Hierarchical Bayesian Model
============================

Models Paul's language as a hierarchical generative process using
Latent Dirichlet Allocation (LDA) with Gibbs sampling.

Theoretical Framework:
    Paul's theological writing is modeled as arising from a three-level
    hierarchical process:

    Level 1 — Core Theology (Latent Topics):
        Abstract theological themes that pervade Paul's thought:
        justification, eschatology, ecclesiology, ethics, Christology, etc.
        These are discovered from the data, not pre-specified.

    Level 2 — Epistle-Specific Applications:
        Each epistle has a distribution over theological topics,
        reflecting how Paul applies his core theology to different
        contexts (Romans emphasizes justification; 1 Corinthians
        addresses ecclesiology and ethics).

    Level 3 — Word Choices (Observed):
        Each topic has a distribution over Paul's vocabulary.
        The actual words Paul writes are drawn from the topic-word
        distributions, weighted by the epistle's topic distribution.

    Generative Process:
        For each epistle d:
            θ_d ~ Dirichlet(α)           # Epistle-topic distribution
            For each word position n in d:
                z_{d,n} ~ Categorical(θ_d)     # Choose topic
                w_{d,n} ~ Categorical(φ_{z})   # Choose word from topic

    Where:
        α = document-topic prior (controls topic mixing)
        β = topic-word prior (controls word specificity)
        φ_k ~ Dirichlet(β) for each topic k

    This is a standard LDA model, but the interpretation is
    specifically tailored to Pauline theology: topics correspond
    to theological themes, and the epistle-topic distributions
    reveal how Paul's theology manifests in different letters.

Key Insight:
    By operating exclusively on Paul's vocabulary, the discovered
    topics represent Paul's own theological categories — not categories
    imposed by external frameworks or other authors' word patterns.

Reference:
    Blei, D.M., Ng, A.Y., & Jordan, M.I. (2003). "Latent Dirichlet
    Allocation." Journal of Machine Learning Research, 3, 993-1022.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..corpus.loader import PaulineCorpus

logger = logging.getLogger(__name__)


@dataclass
class BayesianResult:
    """Results from hierarchical Bayesian topic modeling."""

    # Topic-word distributions: shape (n_topics, vocab_size)
    topic_word_dist: NDArray

    # Document(epistle)-topic distributions: shape (n_docs, n_topics)
    doc_topic_dist: NDArray

    # Vocabulary mapping
    vocabulary: list[str]
    word_to_idx: dict[str, int]

    # Document labels
    doc_labels: list[str]

    # Model parameters
    n_topics: int
    alpha: float
    beta: float
    n_iterations: int

    # Convergence tracking
    log_likelihoods: list[float] = field(default_factory=list)

    @property
    def topic_summaries(self) -> list[dict]:
        """Get top words for each topic."""
        summaries = []
        for k in range(self.n_topics):
            top_indices = np.argsort(self.topic_word_dist[k])[::-1][:15]
            top_words = [
                (self.vocabulary[i], float(self.topic_word_dist[k, i]))
                for i in top_indices
            ]
            summaries.append({
                "topic_id": k,
                "top_words": top_words,
            })
        return summaries

    @property
    def epistle_topic_distributions(self) -> dict[str, list[float]]:
        """Get topic distribution for each epistle."""
        return {
            label: self.doc_topic_dist[i].tolist()
            for i, label in enumerate(self.doc_labels)
        }

    def topic_words(self, topic_id: int, top_n: int = 20) -> list[tuple[str, float]]:
        """Get the top words for a specific topic."""
        if topic_id >= self.n_topics:
            raise ValueError(f"Topic {topic_id} does not exist (n_topics={self.n_topics})")
        top_indices = np.argsort(self.topic_word_dist[topic_id])[::-1][:top_n]
        return [
            (self.vocabulary[i], float(self.topic_word_dist[topic_id, i]))
            for i in top_indices
        ]

    def epistle_topics(self, epistle: str, top_n: int = 5) -> list[tuple[int, float]]:
        """Get the dominant topics for a specific epistle."""
        if epistle not in self.doc_labels:
            raise ValueError(f"Epistle '{epistle}' not found")
        idx = self.doc_labels.index(epistle)
        dist = self.doc_topic_dist[idx]
        top_indices = np.argsort(dist)[::-1][:top_n]
        return [(int(i), float(dist[i])) for i in top_indices]

    def word_topic_affinity(self, word: str) -> list[tuple[int, float]]:
        """Get topic affinities for a specific word (P(topic | word))."""
        if word not in self.word_to_idx:
            return []
        w_idx = self.word_to_idx[word]

        # P(topic | word) ∝ P(word | topic) * P(topic)
        topic_priors = self.doc_topic_dist.mean(axis=0)  # Average topic prevalence
        word_given_topic = self.topic_word_dist[:, w_idx]
        unnormalized = word_given_topic * topic_priors
        normalized = unnormalized / unnormalized.sum()

        return sorted(
            [(int(k), float(normalized[k])) for k in range(self.n_topics)],
            key=lambda x: -x[1],
        )


class HierarchicalBayesianModel:
    """
    Hierarchical Bayesian topic model for the Pauline corpus.

    Implements collapsed Gibbs sampling for LDA, treating each
    epistle as a document. The model discovers latent theological
    topics from Paul's vocabulary distribution patterns.

    Each topic is a probability distribution over Paul's vocabulary,
    and each epistle is a mixture of these topics. The resulting
    structure reveals Paul's latent theological organization.

    Parameters
    ----------
    corpus : PaulineCorpus
        The Pauline corpus to model.
    n_topics : int
        Number of latent topics to discover.
    alpha : float
        Dirichlet prior for document-topic distributions.
        Smaller values → epistles focus on fewer topics.
    beta : float
        Dirichlet prior for topic-word distributions.
        Smaller values → topics use more specific vocabulary.
    """

    def __init__(
        self,
        corpus: PaulineCorpus,
        n_topics: int = 10,
        alpha: float = 0.1,
        beta: float = 0.01,
    ):
        self.corpus = corpus
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

        # Build vocabulary and document-word matrices
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Convert corpus to document-word format for Gibbs sampling."""
        import nltk

        # Build vocabulary from corpus
        word_counts: Counter = Counter()
        for ep in self.corpus.epistles:
            word_counts.update(ep.words)

        # Filter: keep words that appear at least twice and are alphabetic
        self.vocabulary = sorted([
            w for w, c in word_counts.items()
            if c >= 2 and w.isalpha()
        ])
        self.word_to_idx = {w: i for i, w in enumerate(self.vocabulary)}
        self.vocab_size = len(self.vocabulary)

        # Build document data: list of word-index lists
        self.documents: list[list[int]] = []
        self.doc_labels: list[str] = []

        for ep in self.corpus.epistles:
            word_indices = [
                self.word_to_idx[w]
                for w in ep.words
                if w in self.word_to_idx
            ]
            if word_indices:
                self.documents.append(word_indices)
                self.doc_labels.append(ep.name)

        self.n_docs = len(self.documents)
        logger.info(
            f"Prepared Bayesian model: {self.n_docs} documents, "
            f"{self.vocab_size} vocabulary, {self.n_topics} topics"
        )

    def fit(
        self,
        n_iterations: int = 1000,
        seed: Optional[int] = 42,
        log_every: int = 100,
    ) -> BayesianResult:
        """
        Fit the model using collapsed Gibbs sampling.

        Collapsed Gibbs sampling integrates out the topic-word (φ)
        and document-topic (θ) distributions, sampling only the
        topic assignments z for each word. This is more efficient
        and mixes faster than full Gibbs sampling.

        At each iteration, for each word in each document:
            p(z_i = k | z_{-i}, w) ∝
                (n_{d,k}^{-i} + α) * (n_{k,w}^{-i} + β) / (n_{k}^{-i} + V*β)

        Where:
            n_{d,k}^{-i} = count of topic k in document d (excluding i)
            n_{k,w}^{-i} = count of word w assigned to topic k (excluding i)
            n_{k}^{-i}   = total words assigned to topic k (excluding i)
            V = vocabulary size

        Parameters
        ----------
        n_iterations : int
            Number of Gibbs sampling iterations.
        seed : int, optional
            Random seed for reproducibility.
        log_every : int
            Log progress every N iterations.

        Returns
        -------
        BayesianResult
            Fitted model with topic distributions.
        """
        rng = np.random.default_rng(seed)

        K = self.n_topics
        V = self.vocab_size
        D = self.n_docs
        alpha = self.alpha
        beta = self.beta

        # Initialize topic assignments randomly
        z: list[list[int]] = []  # z[d][n] = topic for word n in doc d
        for d in range(D):
            doc_topics = rng.integers(0, K, size=len(self.documents[d])).tolist()
            z.append(doc_topics)

        # Count matrices
        # n_dk[d, k] = number of words in doc d assigned to topic k
        n_dk = np.zeros((D, K), dtype=np.int64)
        # n_kw[k, w] = number of times word w is assigned to topic k
        n_kw = np.zeros((K, V), dtype=np.int64)
        # n_k[k] = total number of words assigned to topic k
        n_k = np.zeros(K, dtype=np.int64)

        # Initialize counts from random assignments
        for d in range(D):
            for n, w in enumerate(self.documents[d]):
                k = z[d][n]
                n_dk[d, k] += 1
                n_kw[k, w] += 1
                n_k[k] += 1

        # Gibbs sampling iterations
        log_likelihoods: list[float] = []

        logger.info(f"Starting Gibbs sampling: {n_iterations} iterations")

        for iteration in range(n_iterations):
            for d in range(D):
                for n, w in enumerate(self.documents[d]):
                    # Remove current assignment
                    k_old = z[d][n]
                    n_dk[d, k_old] -= 1
                    n_kw[k_old, w] -= 1
                    n_k[k_old] -= 1

                    # Compute conditional distribution
                    # p(z=k) ∝ (n_dk[d,k] + α) * (n_kw[k,w] + β) / (n_k[k] + V*β)
                    p = (n_dk[d, :] + alpha) * (n_kw[:, w] + beta) / (n_k + V * beta)
                    p = p / p.sum()

                    # Sample new topic
                    k_new = rng.choice(K, p=p)

                    # Update assignment and counts
                    z[d][n] = k_new
                    n_dk[d, k_new] += 1
                    n_kw[k_new, w] += 1
                    n_k[k_new] += 1

            # Log likelihood (approximate)
            if (iteration + 1) % log_every == 0:
                ll = self._log_likelihood(n_dk, n_kw, n_k, alpha, beta, V)
                log_likelihoods.append(ll)
                logger.info(f"  Iteration {iteration + 1}/{n_iterations}, log-likelihood: {ll:.2f}")

        # Compute final distributions
        # θ_{d,k} = (n_dk[d,k] + α) / (sum_k n_dk[d,k] + K*α)
        doc_topic_dist = (n_dk + alpha).astype(np.float64)
        doc_topic_dist /= doc_topic_dist.sum(axis=1, keepdims=True)

        # φ_{k,w} = (n_kw[k,w] + β) / (n_k[k] + V*β)
        topic_word_dist = (n_kw + beta).astype(np.float64)
        topic_word_dist /= topic_word_dist.sum(axis=1, keepdims=True)

        logger.info("Gibbs sampling complete")

        return BayesianResult(
            topic_word_dist=topic_word_dist,
            doc_topic_dist=doc_topic_dist,
            vocabulary=self.vocabulary,
            word_to_idx=self.word_to_idx,
            doc_labels=self.doc_labels,
            n_topics=self.n_topics,
            alpha=alpha,
            beta=beta,
            n_iterations=n_iterations,
            log_likelihoods=log_likelihoods,
        )

    @staticmethod
    def _log_likelihood(
        n_dk: NDArray, n_kw: NDArray, n_k: NDArray,
        alpha: float, beta: float, V: int,
    ) -> float:
        """
        Compute approximate log-likelihood of the model.

        Uses the standard LDA log-likelihood approximation based on
        the count matrices.
        """
        from scipy.special import gammaln

        D, K = n_dk.shape

        ll = 0.0

        # Document-topic component
        for d in range(D):
            ll += gammaln(K * alpha) - K * gammaln(alpha)
            for k in range(K):
                ll += gammaln(n_dk[d, k] + alpha)
            ll -= gammaln(n_dk[d].sum() + K * alpha)

        # Topic-word component
        for k in range(K):
            ll += gammaln(V * beta) - V * gammaln(beta)
            for w in range(min(V, n_kw.shape[1])):
                ll += gammaln(n_kw[k, w] + beta)
            ll -= gammaln(n_k[k] + V * beta)

        return float(ll)
