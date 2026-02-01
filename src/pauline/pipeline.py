"""
Main Pipeline
=============

Orchestrates the complete Pauline Neural Study pipeline.

Pipeline Phases:
    1. CORPUS    - Fetch/load the Pauline corpus
    2. BOOTSTRAP - Generate bootstrap samples at multiple levels
    3. EMBEDDINGS - Train word embeddings on bootstrap samples
    4. CROSS_EPISTLE - Leave-one-out and all-subsets analysis
    5. COMBINATORIAL - Generate synthetic Pauline text via recombination
    6. FRACTAL - Measure self-similarity at multiple text scales
    7. PERMUTATION - Permutation language modeling on pericopes
    8. VAE - Train variational autoencoder (optional, GPU recommended)
    9. BAYESIAN - Hierarchical Bayesian topic modeling
   10. ANALYSIS - Semantic analysis and visualization

Each phase can be run independently or as part of the full pipeline.
Results are saved to the output directory after each phase.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import PipelineConfig
from .corpus.loader import PaulineCorpus
from .corpus.fetch import CorpusFetcher
from .bootstrap.sampler import BootstrapSampler, SamplingLevel, BootstrapResult
from .embeddings.trainer import EmbeddingTrainer, EmbeddingResult
from .combinatorial.recombiner import CombinatorialRecombiner, RecombinationResult
from .cross_epistle.analyzer import CrossEpistleAnalyzer, CrossEpistleResult

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the complete Pauline Neural Study.

    Usage:
        config = PipelineConfig.from_yaml("configs/default.yaml")
        pipeline = Pipeline(config)
        results = pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.analysis.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline state: populated as phases complete
        self.corpus: Optional[PaulineCorpus] = None
        self.bootstrap_results: dict[SamplingLevel, BootstrapResult] = {}
        self.embedding_results: dict[SamplingLevel, EmbeddingResult] = {}
        self.cross_epistle_result: Optional[CrossEpistleResult] = None
        self.combinatorial_result: Optional[RecombinationResult] = None

    def run(self) -> dict:
        """
        Run all configured pipeline phases.

        Returns:
            Dict of phase_name -> result summary.
        """
        results = {}
        phases = self.config.phases
        total_start = time.time()

        logger.info(f"Starting Pauline Neural Study pipeline")
        logger.info(f"Phases to run: {phases}")
        logger.info(f"Output directory: {self.output_dir}")

        for phase in phases:
            phase_start = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"PHASE: {phase.upper()}")
            logger.info(f"{'='*60}")

            try:
                if phase == "corpus":
                    results[phase] = self._run_corpus()
                elif phase == "bootstrap":
                    results[phase] = self._run_bootstrap()
                elif phase == "embeddings":
                    results[phase] = self._run_embeddings()
                elif phase == "cross_epistle":
                    results[phase] = self._run_cross_epistle()
                elif phase == "combinatorial":
                    results[phase] = self._run_combinatorial()
                elif phase == "fractal":
                    results[phase] = self._run_fractal()
                elif phase == "permutation":
                    results[phase] = self._run_permutation()
                elif phase == "vae":
                    results[phase] = self._run_vae()
                elif phase == "bayesian":
                    results[phase] = self._run_bayesian()
                elif phase == "analysis":
                    results[phase] = self._run_analysis()
                else:
                    logger.warning(f"Unknown phase: {phase}, skipping")
                    continue

                elapsed = time.time() - phase_start
                logger.info(f"Phase {phase} completed in {elapsed:.1f}s")

            except Exception as e:
                logger.error(f"Phase {phase} failed: {e}", exc_info=True)
                results[phase] = {"error": str(e)}

        total_elapsed = time.time() - total_start
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline completed in {total_elapsed:.1f}s")
        logger.info(f"{'='*60}")

        # Save run summary
        self._save_summary(results, total_elapsed)

        return results

    def _run_corpus(self) -> dict:
        """Phase 1: Load or fetch the Pauline corpus."""
        cfg = self.config.corpus

        if cfg.source == "api":
            fetcher = CorpusFetcher(data_dir=cfg.data_dir)
            self.corpus = fetcher.fetch(
                undisputed_only=cfg.undisputed_only,
                translation=cfg.translation,
            )
        elif cfg.source == "json":
            self.corpus = PaulineCorpus.from_json(
                cfg.cache_file,
                undisputed_only=cfg.undisputed_only,
            )
        elif cfg.source == "text_files":
            self.corpus = PaulineCorpus.from_text_files(
                cfg.data_dir,
                undisputed_only=cfg.undisputed_only,
            )
        else:
            raise ValueError(f"Unknown corpus source: {cfg.source}")

        summary = self.corpus.summary()
        logger.info(f"Corpus loaded: {summary['total_words']} words, "
                     f"{summary['vocabulary_size']} unique, "
                     f"{summary['epistles']} epistles")

        # Save corpus summary
        with open(self.output_dir / "corpus_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _run_bootstrap(self) -> dict:
        """Phase 2: Generate bootstrap samples at multiple levels."""
        if self.corpus is None:
            raise RuntimeError("Corpus must be loaded before bootstrap phase")

        cfg = self.config.bootstrap
        sampler = BootstrapSampler(self.corpus, seed=cfg.seed)

        level_map = {
            "epistle": SamplingLevel.EPISTLE,
            "chapter": SamplingLevel.CHAPTER,
            "pericope": SamplingLevel.PERICOPE,
            "sentence": SamplingLevel.SENTENCE,
        }

        results_summary = {}
        for level_name in cfg.levels:
            level = level_map.get(level_name)
            if level is None:
                logger.warning(f"Unknown sampling level: {level_name}")
                continue

            result = sampler.sample(n_samples=cfg.n_samples, level=level)
            self.bootstrap_results[level] = result

            results_summary[level_name] = {
                "n_samples": result.n_samples,
                "avg_words_per_sample": float(result.avg_sample_size),
                "total_words_generated": result.total_words_generated,
            }

        logger.info(f"Bootstrap complete: {len(self.bootstrap_results)} levels sampled")
        return results_summary

    def _run_embeddings(self) -> dict:
        """Phase 3: Train embeddings on bootstrap samples."""
        if not self.bootstrap_results:
            raise RuntimeError("Bootstrap samples must exist before embedding phase")

        cfg = self.config.embedding
        trainer = EmbeddingTrainer(
            embedding_dim=cfg.embedding_dim,
            window=cfg.window,
            min_count=cfg.min_count,
            sg=cfg.sg,
            epochs=cfg.epochs,
            workers=cfg.workers,
            top_n_neighbors=cfg.top_n_neighbors,
        )

        target_words = self.config.analysis.target_words
        results_summary = {}

        for level, bootstrap_result in self.bootstrap_results.items():
            logger.info(f"Training embeddings for {level.value}-level bootstrap...")

            embedding_result = trainer.train_on_bootstrap(
                bootstrap_result,
                target_words=target_words,
            )

            self.embedding_results[level] = embedding_result

            # Save embeddings
            if self.config.analysis.save_embeddings:
                emb_path = self.output_dir / f"embeddings_{level.value}.npz"
                embedding_result.save(emb_path)

            # Log some interesting results
            high_stability = [
                (w, s) for w, s in embedding_result.neighbor_stability.items()
                if s > 0.7
            ]
            high_stability.sort(key=lambda x: -x[1])

            results_summary[level.value] = {
                "vocabulary_size": len(embedding_result.vocabulary),
                "n_samples_trained": embedding_result.n_samples,
                "high_stability_words": len(high_stability),
                "top_stable_words": high_stability[:20],
            }

            logger.info(
                f"  {level.value}: {len(embedding_result.vocabulary)} words, "
                f"{len(high_stability)} with stability > 0.7"
            )

        return results_summary

    def _run_cross_epistle(self) -> dict:
        """Phase 4: Cross-epistle transfer analysis."""
        if self.corpus is None:
            raise RuntimeError("Corpus must be loaded before cross-epistle phase")

        cfg = self.config.cross_epistle
        emb_cfg = self.config.embedding

        analyzer = CrossEpistleAnalyzer(
            corpus=self.corpus,
            embedding_dim=emb_cfg.embedding_dim,
            window=emb_cfg.window,
            min_count=emb_cfg.min_count,
            epochs=emb_cfg.epochs,
        )

        target_words = self.config.analysis.target_words

        if cfg.method == "leave_one_out":
            self.cross_epistle_result = analyzer.leave_one_out(target_words)
        elif cfg.method == "all_subsets":
            self.cross_epistle_result = analyzer.all_subsets(
                target_words, min_subset_size=cfg.min_subset_size
            )
        else:
            raise ValueError(f"Unknown cross-epistle method: {cfg.method}")

        # Extract key findings
        stable = self.cross_epistle_result.stable_relationships()
        variable = self.cross_epistle_result.variable_relationships()

        results_summary = {
            "method": cfg.method,
            "subsets_analyzed": self.cross_epistle_result.epistle_subsets_analyzed,
            "n_stable_relationships": len(stable),
            "n_variable_relationships": len(variable),
            "top_stable": [
                {"words": [r.word1, r.word2], "stability": r.stability}
                for r in stable[:15]
            ],
            "top_variable": [
                {"words": [r.word1, r.word2], "stability": r.stability}
                for r in variable[:15]
            ],
        }

        # Save detailed results
        with open(self.output_dir / "cross_epistle_results.json", "w") as f:
            json.dump(results_summary, f, indent=2)

        return results_summary

    def _run_combinatorial(self) -> dict:
        """Phase 5: Combinatorial recombination."""
        if self.corpus is None:
            raise RuntimeError("Corpus must be loaded before combinatorial phase")

        cfg = self.config.combinatorial
        recombiner = CombinatorialRecombiner(self.corpus, seed=cfg.seed)
        recombiner.prepare()

        # Generate constrained sentences
        self.combinatorial_result = recombiner.generate_constrained(
            n_sentences=cfg.n_sentences,
            bigram_threshold=cfg.bigram_threshold,
            min_length=cfg.min_length,
            max_length=cfg.max_length,
        )

        # Save generated sentences
        with open(self.output_dir / "generated_sentences.txt", "w") as f:
            for sent in self.combinatorial_result.generated_sentences:
                f.write(sent + "\n")

        results_summary = {
            "n_generated": len(self.combinatorial_result.generated_sentences),
            "unique_templates": self.combinatorial_result.unique_templates,
            "vocabulary_used": len(self.combinatorial_result.vocabulary_used),
            "method": self.combinatorial_result.generation_method,
            "constraints": self.combinatorial_result.constraints_applied,
        }

        return results_summary

    def _run_fractal(self) -> dict:
        """Phase 6: Fractal analysis."""
        if self.corpus is None:
            raise RuntimeError("Corpus must be loaded before fractal phase")

        try:
            from .fractal.analyzer import FractalAnalyzer
        except ImportError:
            logger.warning("Fractal analyzer not available, skipping")
            return {"status": "skipped", "reason": "module not available"}

        analyzer = FractalAnalyzer(corpus=self.corpus)

        result = analyzer.analyze()

        results_summary = {
            "hurst_exponent": result.hurst.H,
            "hurst_r_squared": result.hurst.r_squared,
            "dfa_exponent": result.dfa.alpha,
            "dfa_r_squared": result.dfa.r_squared,
            "overall_fractal_score": result.overall_fractal_score,
            "per_epistle": {
                name: {"H": hr.H, "r_squared": hr.r_squared}
                for name, hr in result.epistle_hurst.items()
            },
        }

        with open(self.output_dir / "fractal_results.json", "w") as f:
            json.dump(results_summary, f, indent=2, default=str)

        return results_summary

    def _run_permutation(self) -> dict:
        """Phase 7: Permutation language modeling."""
        if self.corpus is None:
            raise RuntimeError("Corpus must be loaded before permutation phase")

        try:
            from .permutation.plm import PermutationLM
        except ImportError:
            logger.warning("Permutation LM not available, skipping")
            return {"status": "skipped", "reason": "module not available"}

        cfg = self.config.permutation
        plm = PermutationLM(
            corpus=self.corpus,
            seed=cfg.seed,
        )

        result = plm.build_dataset(
            max_perm_per_pericope=cfg.n_permutations_per_unit,
            max_sentences=cfg.max_sentences_per_unit,
        )

        results_summary = {
            "n_pericopes_processed": result.n_pericopes,
            "n_original": result.n_original,
            "n_permuted": result.n_permuted,
            "total_samples": len(result.samples),
            "expansion_factor": result.factorial_expansion,
        }

        return results_summary

    def _run_vae(self) -> dict:
        """Phase 8: Variational autoencoder training."""
        if self.corpus is None:
            raise RuntimeError("Corpus must be loaded before VAE phase")

        try:
            from .vae.model import PaulineVAE
        except ImportError:
            logger.warning("VAE module not available (requires PyTorch), skipping")
            return {"status": "skipped", "reason": "module not available"}

        cfg = self.config.vae
        vae = PaulineVAE(
            corpus=self.corpus,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
        )

        result = vae.train(
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            kl_weight=cfg.kl_weight,
        )

        return {
            "final_loss": result.final_loss,
            "latent_dim": result.latent_dim,
            "vocab_size": result.vocab_size,
            "n_epochs": cfg.n_epochs,
        }

    def _run_bayesian(self) -> dict:
        """Phase 9: Hierarchical Bayesian topic modeling."""
        if self.corpus is None:
            raise RuntimeError("Corpus must be loaded before Bayesian phase")

        try:
            from .bayesian.model import HierarchicalBayesianModel
        except ImportError:
            logger.warning("Bayesian model not available, skipping")
            return {"status": "skipped", "reason": "module not available"}

        cfg = self.config.bayesian
        model = HierarchicalBayesianModel(
            corpus=self.corpus,
            n_topics=cfg.n_topics,
            alpha=cfg.alpha,
            beta=cfg.beta,
        )

        result = model.fit(n_iterations=cfg.n_iterations)

        results_summary = {
            "n_topics": cfg.n_topics,
            "topics": result.topic_summaries,
            "per_epistle_distributions": result.epistle_topic_distributions,
        }

        with open(self.output_dir / "bayesian_topics.json", "w") as f:
            json.dump(results_summary, f, indent=2, default=str)

        return results_summary

    def _run_analysis(self) -> dict:
        """Phase 10: Semantic analysis and visualization."""
        results_summary = {}

        try:
            from .analysis.semantic import SemanticAnalyzer
        except ImportError:
            logger.warning("Semantic analyzer not available, skipping")
            return {"status": "skipped"}

        target_words = self.config.analysis.target_words

        # Analyze embeddings from each level
        for level, emb_result in self.embedding_results.items():
            analyzer = SemanticAnalyzer(
                embedding_result=emb_result,
                cross_epistle_result=self.cross_epistle_result,
            )
            clusters = analyzer.build_semantic_clusters(words=target_words)
            results_summary[f"semantic_{level.value}"] = {
                "n_clusters": len(clusters),
                "clusters": [
                    {"words": c.words, "coherence": c.coherence,
                     "theological_terms": c.theological_terms}
                    for c in clusters
                ],
            }

        # Generate visualizations
        if self.config.analysis.generate_plots:
            try:
                from .analysis.visualization import Visualizer
                viz = Visualizer(output_dir=self.output_dir)

                for level, emb_result in self.embedding_results.items():
                    viz.plot_embedding_space(
                        emb_result,
                        title=f"Pauline Semantic Space ({level.value}-level bootstrap)",
                        filename=f"embedding_space_{level.value}.png",
                    )
                    viz.plot_stability_heatmap(
                        emb_result,
                        title=f"Word Stability ({level.value}-level)",
                        filename=f"stability_{level.value}.png",
                    )

                if self.cross_epistle_result:
                    viz.plot_cross_epistle_influence(
                        self.cross_epistle_result,
                        filename="cross_epistle_influence.png",
                    )

                results_summary["plots_generated"] = True
            except ImportError:
                logger.warning("Visualization module not available, skipping plots")
                results_summary["plots_generated"] = False

        return results_summary

    def _save_summary(self, results: dict, total_elapsed: float) -> None:
        """Save pipeline run summary."""
        summary = {
            "total_elapsed_seconds": total_elapsed,
            "phases_run": list(results.keys()),
            "config": {
                "undisputed_only": self.config.corpus.undisputed_only,
                "n_bootstrap_samples": self.config.bootstrap.n_samples,
                "embedding_dim": self.config.embedding.embedding_dim,
                "n_target_words": len(self.config.analysis.target_words),
            },
            "results": {},
        }

        # Serialize results (handle non-serializable types)
        for phase, result in results.items():
            try:
                json.dumps(result)
                summary["results"][phase] = result
            except (TypeError, ValueError):
                summary["results"][phase] = str(result)

        with open(self.output_dir / "pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Summary saved to {self.output_dir / 'pipeline_summary.json'}")


def main():
    """Entry point for running the pipeline from command line."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Pauline Neural Study Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python -m pauline.pipeline

    # Run with custom config
    python -m pauline.pipeline --config configs/custom.yaml

    # Run specific phases only
    python -m pauline.pipeline --phases corpus bootstrap embeddings analysis

    # Quick test run
    python -m pauline.pipeline --quick
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--phases", "-p",
        nargs="+",
        help="Specific phases to run (overrides config)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test run with reduced samples",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = PipelineConfig.from_yaml(config_path)
    else:
        logger.info(f"Config file {config_path} not found, using defaults")
        config = PipelineConfig()

    # Apply overrides
    if args.phases:
        config.phases = args.phases

    if args.output:
        config.analysis.output_dir = args.output

    if args.quick:
        config.bootstrap.n_samples = 50
        config.embedding.epochs = 10
        config.combinatorial.n_sentences = 50
        logger.info("Quick mode: reduced samples for testing")

    # Ensure NLTK data is available (only needed for English text)
    if config.corpus.source != "text_files" or config.corpus.translation != "greek":
        import nltk
        for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                         "averaged_perceptron_tagger_eng"]:
            try:
                nltk.data.find(
                    f"tokenizers/{resource}" if "punkt" in resource
                    else f"taggers/{resource}"
                )
            except LookupError:
                nltk.download(resource, quiet=True)

    # Run pipeline
    pipeline = Pipeline(config)
    results = pipeline.run()

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for phase, result in results.items():
        if isinstance(result, dict) and "error" in result:
            print(f"  {phase}: FAILED - {result['error']}")
        else:
            print(f"  {phase}: OK")
    print(f"\nResults saved to: {config.analysis.output_dir}/")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
