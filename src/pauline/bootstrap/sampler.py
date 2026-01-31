"""
Bootstrap Sampler
=================

Implements multi-level bootstrap resampling of the Pauline corpus.

Theoretical Foundation (Central Limit Theorem Analogy):
-------------------------------------------------------
Just as the CLT states that the distribution of sample means converges
to the population mean regardless of the population's distribution, we
hypothesize that the distribution of word relationships across bootstrap
samples of Paul's corpus converges to Paul's "true" semantic structure.

Each bootstrap sample creates a valid "alternate Pauline corpus" by
resampling with replacement from Paul's actual text units. By training
word embeddings on hundreds or thousands of these bootstrap samples and
averaging the results, we obtain stable semantic relationships that
represent Paul's authentic usage patterns — without introducing any
external contamination.

Sampling Levels:
    1. Epistle-level: Sample entire epistles with replacement
    2. Chapter-level: Sample chapters with replacement
    3. Pericope-level: Sample theological argument units with replacement
    4. Sentence-level: Sample individual sentences with replacement

The choice of sampling level affects what semantic properties are preserved:
    - Epistle-level preserves macro-theological structure
    - Sentence-level maximizes combinatorial diversity
    - Pericope-level balances theological coherence with diversity

Reference:
    Antoniak, M. & Mimno, D. "Evaluating the Stability of Embedding-based
    Word Similarities" — demonstrates bootstrap sampling for small
    author-specific corpora produces stable, reliable word embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.random import Generator

from ..corpus.loader import PaulineCorpus

logger = logging.getLogger(__name__)


class SamplingLevel(Enum):
    """Granularity level for bootstrap resampling."""
    EPISTLE = "epistle"
    CHAPTER = "chapter"
    PERICOPE = "pericope"
    SENTENCE = "sentence"


@dataclass
class BootstrapSample:
    """A single bootstrap sample of the Pauline corpus."""
    sample_id: int
    level: SamplingLevel
    texts: list[str]  # List of text units in this sample
    source_indices: list[int]  # Which original units were sampled

    @property
    def combined_text(self) -> str:
        return " ".join(self.texts)

    @property
    def words(self) -> list[str]:
        import nltk
        return nltk.word_tokenize(self.combined_text.lower())

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass
class BootstrapResult:
    """Results from a complete bootstrap resampling procedure."""
    level: SamplingLevel
    n_samples: int
    samples: list[BootstrapSample] = field(default_factory=list)
    seed: Optional[int] = None

    @property
    def total_words_generated(self) -> int:
        return sum(s.word_count for s in self.samples)

    @property
    def avg_sample_size(self) -> float:
        if not self.samples:
            return 0.0
        return np.mean([s.word_count for s in self.samples])


class BootstrapSampler:
    """
    Multi-level bootstrap resampler for the Pauline corpus.

    Creates n bootstrap samples by resampling text units with replacement
    at the specified granularity level. Each sample represents a valid
    "alternate Pauline corpus" that preserves Paul's authentic vocabulary
    and usage patterns.

    The CLT analogy: across many bootstrap samples, word co-occurrence
    statistics converge to their "true" values in Paul's semantic system,
    even though each individual sample is a partial view.
    """

    def __init__(self, corpus: PaulineCorpus, seed: Optional[int] = None):
        self.corpus = corpus
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._prepare_units()

    def _prepare_units(self) -> None:
        """Pre-compute text units at each sampling level."""
        import nltk

        # Epistle-level units
        self._epistle_units: list[str] = [
            ep.text for ep in self.corpus.epistles
        ]
        self._epistle_labels: list[str] = [
            ep.name for ep in self.corpus.epistles
        ]

        # Chapter-level units
        self._chapter_units: list[str] = []
        self._chapter_labels: list[str] = []
        for ep in self.corpus.epistles:
            for ch_num, verses in ep.chapters.items():
                ch_text = " ".join(v.text for v in verses)
                self._chapter_units.append(ch_text)
                self._chapter_labels.append(f"{ep.abbreviation} {ch_num}")

        # Pericope-level units (if available, otherwise fall back to chapters)
        self._pericope_units: list[str] = []
        self._pericope_labels: list[str] = []
        for ep in self.corpus.epistles:
            if ep.pericopes:
                for peri in ep.pericopes:
                    self._pericope_units.append(peri.text)
                    self._pericope_labels.append(f"{ep.abbreviation}: {peri.label}")
            else:
                # Fall back to chapters as pseudo-pericopes
                for ch_num, verses in ep.chapters.items():
                    ch_text = " ".join(v.text for v in verses)
                    self._pericope_units.append(ch_text)
                    self._pericope_labels.append(f"{ep.abbreviation} {ch_num}")

        # Sentence-level units
        self._sentence_units: list[str] = [
            sent for _, sent in self.corpus.all_sentences
        ]
        self._sentence_labels: list[str] = [
            book for book, _ in self.corpus.all_sentences
        ]

        logger.info(
            f"Prepared sampling units: "
            f"{len(self._epistle_units)} epistles, "
            f"{len(self._chapter_units)} chapters, "
            f"{len(self._pericope_units)} pericopes, "
            f"{len(self._sentence_units)} sentences"
        )

    def _get_units(self, level: SamplingLevel) -> tuple[list[str], list[str]]:
        """Get text units and labels for the specified level."""
        if level == SamplingLevel.EPISTLE:
            return self._epistle_units, self._epistle_labels
        elif level == SamplingLevel.CHAPTER:
            return self._chapter_units, self._chapter_labels
        elif level == SamplingLevel.PERICOPE:
            return self._pericope_units, self._pericope_labels
        elif level == SamplingLevel.SENTENCE:
            return self._sentence_units, self._sentence_labels
        else:
            raise ValueError(f"Unknown sampling level: {level}")

    def sample(
        self,
        n_samples: int = 1000,
        level: SamplingLevel = SamplingLevel.SENTENCE,
        sample_size: Optional[int] = None,
    ) -> BootstrapResult:
        """
        Generate bootstrap samples at the specified level.

        Args:
            n_samples: Number of bootstrap samples to generate.
            level: Granularity level for resampling.
            sample_size: Number of units per sample. If None, uses the
                        same size as the original (standard bootstrap).

        Returns:
            BootstrapResult containing all samples with metadata.

        Theory:
            Standard bootstrap: sample n units with replacement from
            n original units. Each bootstrap sample has the same size
            as the original corpus but with some units duplicated and
            others omitted. This is the statistical basis for
            estimating sampling distributions from a single dataset.
        """
        units, labels = self._get_units(level)
        n_units = len(units)

        if sample_size is None:
            sample_size = n_units

        logger.info(
            f"Generating {n_samples} bootstrap samples at {level.value} level "
            f"({n_units} original units, {sample_size} units per sample)"
        )

        result = BootstrapResult(
            level=level,
            n_samples=n_samples,
            seed=self.seed,
        )

        for i in range(n_samples):
            # Sample with replacement
            indices = self.rng.integers(0, n_units, size=sample_size)
            sampled_texts = [units[idx] for idx in indices]

            result.samples.append(BootstrapSample(
                sample_id=i,
                level=level,
                texts=sampled_texts,
                source_indices=indices.tolist(),
            ))

            if (i + 1) % 100 == 0:
                logger.debug(f"  Generated {i + 1}/{n_samples} samples")

        logger.info(
            f"Bootstrap complete: {n_samples} samples, "
            f"~{result.avg_sample_size:.0f} words per sample, "
            f"{result.total_words_generated:,} total words generated"
        )

        return result

    def multi_level_sample(
        self,
        n_samples: int = 500,
        levels: Optional[list[SamplingLevel]] = None,
    ) -> dict[SamplingLevel, BootstrapResult]:
        """
        Generate bootstrap samples at multiple levels simultaneously.

        This allows comparing semantic stability across different
        granularity levels, revealing which word relationships are
        robust at all levels (true Pauline semantics) versus those
        that only appear at certain granularities (context-dependent).

        Args:
            n_samples: Number of samples per level.
            levels: Which levels to sample. Default: all levels.

        Returns:
            Dict mapping each level to its BootstrapResult.
        """
        if levels is None:
            levels = [
                SamplingLevel.EPISTLE,
                SamplingLevel.CHAPTER,
                SamplingLevel.SENTENCE,
            ]

        results = {}
        for level in levels:
            results[level] = self.sample(n_samples=n_samples, level=level)

        return results

    def jackknife_epistles(self) -> list[tuple[str, list[str]]]:
        """
        Leave-one-out jackknife at the epistle level.

        For each epistle, creates a sub-corpus containing all OTHER
        epistles. This is used by the cross-epistle analyzer to measure
        how word relationships change when specific epistles are removed.

        Returns:
            List of (left_out_epistle_name, remaining_texts) tuples.
        """
        units = self._epistle_units
        labels = self._epistle_labels

        jackknife_samples = []
        for i in range(len(units)):
            remaining = [units[j] for j in range(len(units)) if j != i]
            jackknife_samples.append((labels[i], remaining))

        return jackknife_samples
