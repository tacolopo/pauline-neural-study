"""
Fractal / Self-Similarity Analyzer
===================================

Analyzes the Pauline corpus for fractal-like self-similarity across
multiple textual scales: word co-occurrence within sentences, sentence
relationships within pericopes, and pericope flow within epistles.

Theoretical Foundation:
-----------------------
Natural language exhibits scale-free properties analogous to fractals in
geometry. Just as a fractal coastline shows similar complexity whether
measured at 1 km or 100 km resolution, a coherent author's writing may
exhibit consistent statistical patterns at word, sentence, pericope, and
epistle scales. If Paul's theological reasoning is internally consistent,
we expect self-similar distributional signatures at every scale.

This module measures that self-similarity using three complementary
techniques:

1. **Hurst Exponent (H)**: Quantifies long-range dependence in a
   time series of word features. For Pauline text, we convert the
   sequential token stream into a numeric series (e.g., word frequency
   ranks, semantic novelty scores) and estimate H via rescaled-range
   (R/S) analysis. H > 0.5 indicates persistent long-range correlations
   (the "Pauline voice" carries across passages); H = 0.5 is random;
   H < 0.5 is anti-persistent.

2. **Detrended Fluctuation Analysis (DFA)**: A more robust alternative
   to R/S analysis that removes polynomial trends before measuring
   fluctuations. The DFA exponent alpha has the same interpretation as H
   but is less sensitive to non-stationarity (important because Paul's
   vocabulary shifts across topics).

3. **Multi-Scale Co-occurrence Similarity**: Measures how word
   co-occurrence matrices computed at different textual scales
   (sentence, pericope, epistle) correlate with each other. High
   cross-scale correlation indicates genuine fractal self-similarity
   in Paul's semantic structure.

References:
    - Mandelbrot, B. "The Fractal Geometry of Nature" (1982)
    - Peng, C.K. et al. "Mosaic organization of DNA nucleotides" (1994)
      — introduces DFA
    - Altmann, E.G. & Gerlach, M. "Statistical laws in linguistics" (2016)
    - Montemurro, M.A. & Zanette, D.H. "Towards the quantification of the
      semantic information encoded in written language" (2010)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial.distance import cosine

import nltk

from ..corpus.loader import PaulineCorpus, Epistle, Pericope

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HurstResult:
    """Result from Hurst exponent estimation via R/S analysis.

    Attributes:
        H: Estimated Hurst exponent in [0, 1].
        intercept: Intercept of the log-log regression.
        r_squared: Goodness-of-fit of the log-log regression.
        log_ns: Array of log(window sizes) used.
        log_rs: Array of log(rescaled ranges) computed.
        interpretation: Human-readable interpretation string.
    """
    H: float
    intercept: float
    r_squared: float
    log_ns: NDArray
    log_rs: NDArray
    interpretation: str

    @staticmethod
    def interpret(h: float) -> str:
        """Provide a textual interpretation of the Hurst exponent."""
        if h > 0.65:
            return (
                f"H = {h:.3f}: Strong long-range persistence. Paul's "
                "stylistic/thematic patterns carry forward across large "
                "stretches of text — consistent authorial voice."
            )
        elif h > 0.55:
            return (
                f"H = {h:.3f}: Moderate persistence. Pauline style shows "
                "meaningful long-range correlations beyond what random "
                "word shuffling would produce."
            )
        elif h > 0.45:
            return (
                f"H = {h:.3f}: Near-random (Brownian). The measured "
                "feature behaves close to a random walk at this scale."
            )
        else:
            return (
                f"H = {h:.3f}: Anti-persistent. Successive values tend "
                "to reverse direction — possibly reflecting Paul's "
                "dialectical argument style (thesis-antithesis)."
            )


@dataclass
class DFAResult:
    """Result from Detrended Fluctuation Analysis.

    Attributes:
        alpha: DFA scaling exponent. Interpretation similar to Hurst H.
        intercept: Intercept of the log-log regression.
        r_squared: Goodness-of-fit.
        log_ns: Array of log(box sizes) used.
        log_fluct: Array of log(fluctuation) values.
        interpretation: Human-readable interpretation string.
    """
    alpha: float
    intercept: float
    r_squared: float
    log_ns: NDArray
    log_fluct: NDArray
    interpretation: str

    @staticmethod
    def interpret(alpha: float) -> str:
        """Provide a textual interpretation of the DFA exponent."""
        if alpha > 1.0:
            return (
                f"alpha = {alpha:.3f}: Non-stationary, strong long-range "
                "correlations (1/f noise regime). Pauline discourse "
                "structure shows deep, persistent organization."
            )
        elif alpha > 0.65:
            return (
                f"alpha = {alpha:.3f}: Long-range correlated. Paul's word "
                "patterns show memory effects that span many sentences."
            )
        elif alpha > 0.55:
            return (
                f"alpha = {alpha:.3f}: Weakly correlated, close to the "
                "1/f boundary. Some long-range structure is present."
            )
        else:
            return (
                f"alpha = {alpha:.3f}: White-noise-like at this scale. "
                "The measured feature lacks long-range correlations."
            )


@dataclass
class MultiScaleResult:
    """Result from multi-scale co-occurrence similarity analysis.

    Attributes:
        scale_names: Labels for each scale analysed.
        cooccurrence_matrices: Dict mapping scale name to its
            co-occurrence matrix (vocab_size x vocab_size).
        cross_scale_correlations: Dict mapping (scale_a, scale_b) to
            the Pearson correlation of their flattened upper-triangular
            co-occurrence values.
        fractal_dimension_estimate: Rough box-counting fractal dimension
            estimated from the co-occurrence graph at multiple thresholds.
        vocabulary: The shared vocabulary across all scales.
    """
    scale_names: list[str]
    cooccurrence_matrices: dict[str, NDArray]
    cross_scale_correlations: dict[tuple[str, str], float]
    fractal_dimension_estimate: float
    vocabulary: list[str]


@dataclass
class FractalResult:
    """Aggregated results from all fractal analyses.

    Attributes:
        hurst: Hurst exponent analysis result.
        dfa: DFA analysis result.
        multi_scale: Multi-scale co-occurrence result.
        epistle_hurst: Per-epistle Hurst exponent results.
        overall_fractal_score: Composite self-similarity score in [0, 1].
    """
    hurst: HurstResult
    dfa: DFAResult
    multi_scale: Optional[MultiScaleResult] = None
    epistle_hurst: dict[str, HurstResult] = field(default_factory=dict)
    overall_fractal_score: float = 0.0


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class FractalAnalyzer:
    """
    Multi-scale fractal and self-similarity analysis of Pauline text.

    Converts the sequential text stream into numeric feature series and
    applies long-range dependence estimators (Hurst exponent, DFA) as
    well as cross-scale co-occurrence analysis to quantify fractal
    structure in Paul's writing.

    Usage::

        from pauline.corpus.loader import PaulineCorpus
        from pauline.fractal.analyzer import FractalAnalyzer

        corpus = PaulineCorpus.from_json("pauline_corpus.json")
        analyzer = FractalAnalyzer(corpus)
        result = analyzer.analyze()

        print(result.hurst.interpretation)
        print(result.dfa.interpretation)
        print(f"Fractal score: {result.overall_fractal_score:.3f}")
    """

    # Minimum series length for reliable estimation
    MIN_SERIES_LENGTH = 64

    def __init__(
        self,
        corpus: PaulineCorpus,
        seed: Optional[int] = None,
    ):
        self.corpus = corpus
        self.rng = np.random.default_rng(seed)
        self._word_freq: dict[str, int] = corpus.word_frequency()
        self._vocabulary: list[str] = sorted(self._word_freq.keys())
        self._word_to_rank: dict[str, int] = {
            w: i for i, w in enumerate(self._vocabulary)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        feature: str = "frequency_rank",
        dfa_orders: list[int] | None = None,
        multi_scale: bool = True,
        per_epistle: bool = True,
    ) -> FractalResult:
        """
        Run full fractal analysis pipeline.

        Args:
            feature: Which numeric feature to derive from the token
                stream. Options:
                - ``"frequency_rank"``: each token mapped to its
                  corpus-wide frequency rank (most common = 0).
                - ``"log_frequency"``: log(1 + raw frequency count).
                - ``"novelty"``: binary indicator for first occurrence
                  within a sliding window (measures lexical surprise).
            dfa_orders: Polynomial detrending orders for DFA. Default
                ``[1, 2]`` (linear and quadratic detrending).
            multi_scale: Whether to compute multi-scale co-occurrence.
            per_epistle: Whether to compute per-epistle Hurst exponents.

        Returns:
            FractalResult with all computed metrics.
        """
        if dfa_orders is None:
            dfa_orders = [1, 2]

        # Build the corpus-wide feature series
        series = self._text_to_series(self.corpus.all_words, feature)

        if len(series) < self.MIN_SERIES_LENGTH:
            raise ValueError(
                f"Corpus too short ({len(series)} tokens) for reliable "
                f"fractal analysis (need >= {self.MIN_SERIES_LENGTH})."
            )

        logger.info(
            f"Fractal analysis: {len(series)} tokens, "
            f"feature='{feature}', DFA orders={dfa_orders}"
        )

        # Hurst exponent (R/S analysis)
        hurst = self.hurst_exponent(series)

        # DFA (use highest-order result)
        dfa_results = [self.dfa(series, order=o) for o in dfa_orders]
        best_dfa = max(dfa_results, key=lambda r: r.r_squared)

        # Multi-scale co-occurrence
        ms_result = None
        if multi_scale:
            ms_result = self.multi_scale_cooccurrence()

        # Per-epistle Hurst
        epistle_hurst: dict[str, HurstResult] = {}
        if per_epistle:
            for ep in self.corpus.epistles:
                ep_series = self._text_to_series(ep.words, feature)
                if len(ep_series) >= self.MIN_SERIES_LENGTH:
                    epistle_hurst[ep.name] = self.hurst_exponent(ep_series)
                else:
                    logger.debug(
                        f"Skipping {ep.name}: too short "
                        f"({len(ep_series)} tokens)"
                    )

        # Composite fractal score
        score = self._composite_score(hurst, best_dfa, ms_result)

        return FractalResult(
            hurst=hurst,
            dfa=best_dfa,
            multi_scale=ms_result,
            epistle_hurst=epistle_hurst,
            overall_fractal_score=score,
        )

    # ------------------------------------------------------------------
    # Hurst exponent via Rescaled-Range (R/S) analysis
    # ------------------------------------------------------------------

    def hurst_exponent(
        self,
        series: NDArray,
        min_window: int = 8,
    ) -> HurstResult:
        """
        Estimate the Hurst exponent via rescaled-range (R/S) analysis.

        The R/S statistic for window size n is:
            R(n) / S(n)
        where R(n) is the range of the cumulative deviation from the
        mean over the window, and S(n) is the standard deviation.

        For a self-similar process, E[R/S] ~ C * n^H, so H is the
        slope of log(R/S) vs log(n).

        Args:
            series: 1-D numeric array (the feature time series).
            min_window: Smallest window size to use.

        Returns:
            HurstResult with estimated H and diagnostics.
        """
        n = len(series)
        max_window = n // 4

        if max_window < min_window:
            min_window = max(4, max_window // 2)

        # Generate window sizes (powers of 2 and intermediate points)
        window_sizes = []
        size = min_window
        while size <= max_window:
            window_sizes.append(size)
            size = int(size * 1.5)
            if size == window_sizes[-1]:
                size += 1

        if len(window_sizes) < 3:
            # Fall back to a simple linear spacing
            window_sizes = list(
                range(min_window, max_window + 1, max(1, (max_window - min_window) // 8))
            )

        log_ns = []
        log_rs = []

        for ws in window_sizes:
            if ws < 4:
                continue
            rs_values = []
            n_segments = n // ws
            if n_segments < 1:
                continue

            for seg_idx in range(n_segments):
                segment = series[seg_idx * ws : (seg_idx + 1) * ws]
                mean_seg = np.mean(segment)
                cumulative_dev = np.cumsum(segment - mean_seg)
                r = np.max(cumulative_dev) - np.min(cumulative_dev)
                s = np.std(segment, ddof=1)
                if s > 1e-12:
                    rs_values.append(r / s)

            if rs_values:
                log_ns.append(np.log(ws))
                log_rs.append(np.log(np.mean(rs_values)))

        log_ns = np.array(log_ns)
        log_rs = np.array(log_rs)

        if len(log_ns) < 2:
            logger.warning(
                "Too few valid window sizes for Hurst estimation; "
                "returning H = 0.5 (indeterminate)"
            )
            return HurstResult(
                H=0.5,
                intercept=0.0,
                r_squared=0.0,
                log_ns=log_ns,
                log_rs=log_rs,
                interpretation=HurstResult.interpret(0.5),
            )

        slope, intercept, r_value, _, _ = stats.linregress(log_ns, log_rs)
        h = float(np.clip(slope, 0.0, 1.0))
        r_sq = r_value ** 2

        return HurstResult(
            H=h,
            intercept=float(intercept),
            r_squared=float(r_sq),
            log_ns=log_ns,
            log_rs=log_rs,
            interpretation=HurstResult.interpret(h),
        )

    # ------------------------------------------------------------------
    # Detrended Fluctuation Analysis (DFA)
    # ------------------------------------------------------------------

    def dfa(
        self,
        series: NDArray,
        order: int = 1,
        min_box: int = 8,
    ) -> DFAResult:
        """
        Detrended Fluctuation Analysis.

        Steps:
            1. Integrate the mean-subtracted series: Y(k) = sum(x_i - x_bar).
            2. Divide Y into non-overlapping boxes of size n.
            3. In each box, fit a polynomial of degree ``order`` (the local
               trend) and compute the residual variance.
            4. Average the residual variance across boxes to get F(n).
            5. The DFA exponent alpha is the slope of log(F) vs log(n).

        Args:
            series: 1-D numeric feature series.
            order: Polynomial detrending order (1 = linear, 2 = quadratic).
            min_box: Smallest box size.

        Returns:
            DFAResult with estimated alpha and diagnostics.
        """
        n = len(series)
        # Integrate the mean-subtracted series
        y = np.cumsum(series - np.mean(series))

        max_box = n // 4
        if max_box < min_box:
            min_box = max(4, max_box // 2)

        # Box sizes: logarithmically spaced
        box_sizes = np.unique(
            np.logspace(
                np.log10(min_box),
                np.log10(max(max_box, min_box + 1)),
                num=20,
            ).astype(int)
        )
        box_sizes = box_sizes[box_sizes >= 4]

        if len(box_sizes) < 3:
            box_sizes = np.arange(min_box, max(max_box + 1, min_box + 4))

        log_ns = []
        log_fluct = []

        for bs in box_sizes:
            bs = int(bs)
            n_boxes = n // bs
            if n_boxes < 1:
                continue

            variances = []
            for b in range(n_boxes):
                segment = y[b * bs : (b + 1) * bs]
                # Fit polynomial trend
                x_range = np.arange(bs)
                coeffs = np.polyfit(x_range, segment, deg=order)
                trend = np.polyval(coeffs, x_range)
                residual = segment - trend
                variances.append(np.mean(residual ** 2))

            if variances:
                fluct = np.sqrt(np.mean(variances))
                if fluct > 1e-12:
                    log_ns.append(np.log(bs))
                    log_fluct.append(np.log(fluct))

        log_ns = np.array(log_ns)
        log_fluct = np.array(log_fluct)

        if len(log_ns) < 2:
            logger.warning(
                "Too few valid box sizes for DFA; returning alpha = 0.5"
            )
            return DFAResult(
                alpha=0.5,
                intercept=0.0,
                r_squared=0.0,
                log_ns=log_ns,
                log_fluct=log_fluct,
                interpretation=DFAResult.interpret(0.5),
            )

        slope, intercept, r_value, _, _ = stats.linregress(log_ns, log_fluct)
        alpha = float(slope)
        r_sq = r_value ** 2

        return DFAResult(
            alpha=alpha,
            intercept=float(intercept),
            r_squared=float(r_sq),
            log_ns=log_ns,
            log_fluct=log_fluct,
            interpretation=DFAResult.interpret(alpha),
        )

    # ------------------------------------------------------------------
    # Multi-scale co-occurrence analysis
    # ------------------------------------------------------------------

    def multi_scale_cooccurrence(
        self,
        top_n_vocab: int = 200,
        threshold_steps: int = 10,
    ) -> MultiScaleResult:
        """
        Compute word co-occurrence matrices at three textual scales and
        measure cross-scale correlation.

        Scales:
            1. **Sentence**: words co-occur within the same sentence.
            2. **Pericope**: words co-occur within the same pericope
               (theological argument unit).
            3. **Epistle**: words co-occur within the same epistle.

        If Paul's writing is fractal / self-similar, co-occurrence
        patterns at fine scales (sentences) should correlate with
        patterns at coarse scales (epistles).

        Args:
            top_n_vocab: Number of most frequent words to include.
            threshold_steps: Number of thresholds for box-counting
                fractal dimension estimate.

        Returns:
            MultiScaleResult with matrices, correlations, and fractal
            dimension estimate.
        """
        # Build restricted vocabulary (top N by frequency)
        sorted_words = sorted(
            self._word_freq.items(), key=lambda x: -x[1]
        )
        # Keep only alphabetic tokens
        vocab = [
            w for w, _ in sorted_words
            if w.isalpha()
        ][:top_n_vocab]
        vocab_idx = {w: i for i, w in enumerate(vocab)}
        v = len(vocab)

        logger.info(
            f"Multi-scale co-occurrence: {v} vocabulary items, "
            f"3 scales (sentence, pericope, epistle)"
        )

        # --- Sentence-scale ---
        sent_matrix = np.zeros((v, v), dtype=np.float64)
        for _, sent_text in self.corpus.all_sentences:
            tokens = set(nltk.word_tokenize(sent_text.lower()))
            present = [vocab_idx[t] for t in tokens if t in vocab_idx]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    sent_matrix[present[i], present[j]] += 1
                    sent_matrix[present[j], present[i]] += 1

        # --- Pericope-scale ---
        peri_matrix = np.zeros((v, v), dtype=np.float64)
        pericopes_found = 0
        for ep in self.corpus.epistles:
            units: list[str] = []
            if ep.pericopes:
                units = [p.text for p in ep.pericopes]
            else:
                # Fall back to chapters as pseudo-pericopes
                for _, verses in ep.chapters.items():
                    units.append(" ".join(vv.text for vv in verses))
            pericopes_found += len(units)

            for unit_text in units:
                tokens = set(nltk.word_tokenize(unit_text.lower()))
                present = [vocab_idx[t] for t in tokens if t in vocab_idx]
                for i in range(len(present)):
                    for j in range(i + 1, len(present)):
                        peri_matrix[present[i], present[j]] += 1
                        peri_matrix[present[j], present[i]] += 1

        logger.debug(f"Pericope/chapter units found: {pericopes_found}")

        # --- Epistle-scale ---
        ep_matrix = np.zeros((v, v), dtype=np.float64)
        for ep in self.corpus.epistles:
            tokens = set(ep.words)
            present = [vocab_idx[t] for t in tokens if t in vocab_idx]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    ep_matrix[present[i], present[j]] += 1
                    ep_matrix[present[j], present[i]] += 1

        # Normalize matrices to [0, 1]
        for mat in (sent_matrix, peri_matrix, ep_matrix):
            mat_max = mat.max()
            if mat_max > 0:
                mat /= mat_max

        matrices = {
            "sentence": sent_matrix,
            "pericope": peri_matrix,
            "epistle": ep_matrix,
        }
        scale_names = list(matrices.keys())

        # Cross-scale correlations (upper-triangle Pearson r)
        cross_corr: dict[tuple[str, str], float] = {}
        triu_idx = np.triu_indices(v, k=1)

        for i in range(len(scale_names)):
            for j in range(i + 1, len(scale_names)):
                sa, sb = scale_names[i], scale_names[j]
                vec_a = matrices[sa][triu_idx]
                vec_b = matrices[sb][triu_idx]
                if np.std(vec_a) > 1e-12 and np.std(vec_b) > 1e-12:
                    r, _ = stats.pearsonr(vec_a, vec_b)
                    cross_corr[(sa, sb)] = float(r)
                else:
                    cross_corr[(sa, sb)] = 0.0

        # Box-counting fractal dimension estimate on sentence co-occurrence
        fd = self._box_counting_dimension(sent_matrix, threshold_steps)

        return MultiScaleResult(
            scale_names=scale_names,
            cooccurrence_matrices=matrices,
            cross_scale_correlations=cross_corr,
            fractal_dimension_estimate=fd,
            vocabulary=vocab,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _text_to_series(
        self,
        words: list[str],
        feature: str,
    ) -> NDArray:
        """
        Convert a token sequence to a numeric feature series.

        Args:
            words: Sequential token list.
            feature: Feature type (see ``analyze`` docstring).

        Returns:
            1-D float array of the same length as ``words``.
        """
        if feature == "frequency_rank":
            max_rank = len(self._vocabulary)
            series = np.array([
                self._word_to_rank.get(w, max_rank)
                for w in words
            ], dtype=np.float64)
            # Normalize to [0, 1]
            if series.max() > 0:
                series /= series.max()

        elif feature == "log_frequency":
            series = np.array([
                np.log1p(self._word_freq.get(w, 0))
                for w in words
            ], dtype=np.float64)

        elif feature == "novelty":
            # 1.0 if the word has not appeared in the preceding window
            window = 50
            seen: dict[str, int] = {}
            vals = []
            for i, w in enumerate(words):
                last_seen = seen.get(w, -window - 1)
                if i - last_seen > window:
                    vals.append(1.0)
                else:
                    vals.append(0.0)
                seen[w] = i
            series = np.array(vals, dtype=np.float64)

        else:
            raise ValueError(
                f"Unknown feature '{feature}'. "
                f"Choose from: frequency_rank, log_frequency, novelty"
            )

        return series

    def _box_counting_dimension(
        self,
        matrix: NDArray,
        n_thresholds: int = 10,
    ) -> float:
        """
        Estimate the fractal dimension of a co-occurrence graph via
        box counting.

        Treat the co-occurrence matrix as a weighted adjacency matrix.
        At each threshold t, binarize (edge exists if weight >= t) and
        count the number of non-empty "boxes" (connected components or,
        more simply, the number of non-zero entries).

        The fractal dimension D_b is estimated from the slope of
        log(N_boxes) vs log(1/threshold).

        Args:
            matrix: Normalized co-occurrence matrix.
            n_thresholds: Number of threshold levels.

        Returns:
            Estimated box-counting fractal dimension.
        """
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        log_inv_t = []
        log_count = []

        for t in thresholds:
            binary = (matrix >= t).astype(int)
            count = np.sum(binary)
            if count > 0:
                log_inv_t.append(np.log(1.0 / t))
                log_count.append(np.log(count))

        if len(log_inv_t) < 3:
            return 0.0

        log_inv_t = np.array(log_inv_t)
        log_count = np.array(log_count)

        slope, _, _, _, _ = stats.linregress(log_inv_t, log_count)
        return float(slope)

    def _composite_score(
        self,
        hurst: HurstResult,
        dfa: DFAResult,
        multi_scale: Optional[MultiScaleResult],
    ) -> float:
        """
        Compute a composite fractal / self-similarity score in [0, 1].

        Components (weighted average):
            - Hurst persistence strength: |H - 0.5| * 2  (weight 0.3)
            - DFA persistence strength: clip(alpha - 0.5, 0, 1) (weight 0.3)
            - Mean cross-scale correlation                 (weight 0.4)

        A score near 1.0 means strong, consistent self-similarity across
        all metrics; near 0.0 means the text is indistinguishable from
        a random token sequence.
        """
        # Hurst contribution: how far from random (0.5)?
        h_score = min(abs(hurst.H - 0.5) * 2.0, 1.0)

        # DFA contribution: alpha > 0.5 indicates correlations
        dfa_score = float(np.clip(dfa.alpha - 0.5, 0.0, 1.0))

        # Cross-scale correlation contribution
        if multi_scale and multi_scale.cross_scale_correlations:
            corr_vals = list(multi_scale.cross_scale_correlations.values())
            ms_score = float(np.clip(np.mean(corr_vals), 0.0, 1.0))
        else:
            ms_score = 0.0

        composite = 0.3 * h_score + 0.3 * dfa_score + 0.4 * ms_score
        return float(np.clip(composite, 0.0, 1.0))
