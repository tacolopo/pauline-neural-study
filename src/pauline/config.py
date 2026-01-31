"""
Configuration
=============

Central configuration for the Pauline Neural Study pipeline.
Loads from YAML config files with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class CorpusConfig:
    """Corpus loading configuration."""
    data_dir: str = "data"
    undisputed_only: bool = True
    source: str = "api"  # "api", "json", "text_files"
    translation: str = "web"
    cache_file: str = "data/pauline_corpus.json"


@dataclass
class BootstrapConfig:
    """Bootstrap sampling configuration."""
    n_samples: int = 1000
    levels: list[str] = field(default_factory=lambda: ["sentence", "chapter", "epistle"])
    seed: int = 42


@dataclass
class EmbeddingConfig:
    """Word embedding training configuration."""
    embedding_dim: int = 100
    window: int = 5
    min_count: int = 2
    sg: int = 1  # Skip-gram
    epochs: int = 50
    workers: int = 4
    top_n_neighbors: int = 10


@dataclass
class CombinatorialConfig:
    """Combinatorial recombination configuration."""
    n_sentences: int = 500
    min_length: int = 5
    max_length: int = 30
    bigram_threshold: int = 1
    seed: int = 42


@dataclass
class CrossEpistleConfig:
    """Cross-epistle analysis configuration."""
    method: str = "leave_one_out"  # "leave_one_out" or "all_subsets"
    min_subset_size: int = 3


@dataclass
class FractalConfig:
    """Fractal analysis configuration."""
    min_window: int = 4
    max_window: int = 256
    n_windows: int = 20


@dataclass
class PermutationConfig:
    """Permutation language modeling configuration."""
    max_sentences_per_unit: int = 8
    n_permutations_per_unit: int = 20
    seed: int = 42


@dataclass
class VAEConfig:
    """Variational autoencoder configuration."""
    latent_dim: int = 64
    hidden_dim: int = 256
    max_seq_length: int = 50
    batch_size: int = 32
    n_epochs: int = 100
    learning_rate: float = 0.001
    kl_weight: float = 0.5


@dataclass
class BayesianConfig:
    """Hierarchical Bayesian model configuration."""
    n_topics: int = 10
    n_iterations: int = 1000
    alpha: float = 0.1
    beta: float = 0.01


@dataclass
class AnalysisConfig:
    """Analysis and output configuration."""
    output_dir: str = "output"
    target_words: list[str] = field(default_factory=lambda: [
        # Core theological terms (English equivalents for WEB text)
        "justification", "justified", "justify", "righteous", "righteousness",
        "faith", "believe", "belief",
        "law", "commandment", "commandments",
        "grace", "mercy", "gift",
        "sin", "sins", "sinned", "sinner",
        "spirit", "flesh", "body",
        "christ", "jesus", "lord", "god",
        "love", "hope", "peace",
        "death", "life", "resurrection",
        "cross", "blood", "sacrifice",
        "church", "body", "members",
        "salvation", "saved", "save",
        "gospel", "preach", "preaching",
        "circumcision", "uncircumcision",
        "gentile", "gentiles", "jew", "jews",
        "abraham", "moses",
        "covenant", "promise", "promises",
        "glory", "power", "wisdom",
        "freedom", "slave", "bondage",
        "baptism", "baptized",
    ])
    generate_plots: bool = True
    save_embeddings: bool = True


@dataclass
class PipelineConfig:
    """Master configuration for the full pipeline."""
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    combinatorial: CombinatorialConfig = field(default_factory=CombinatorialConfig)
    cross_epistle: CrossEpistleConfig = field(default_factory=CrossEpistleConfig)
    fractal: FractalConfig = field(default_factory=FractalConfig)
    permutation: PermutationConfig = field(default_factory=PermutationConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # Pipeline control: which phases to run
    phases: list[str] = field(default_factory=lambda: [
        "corpus",
        "bootstrap",
        "embeddings",
        "cross_epistle",
        "combinatorial",
        "fractal",
        "analysis",
    ])

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        if "corpus" in data:
            config.corpus = CorpusConfig(**data["corpus"])
        if "bootstrap" in data:
            config.bootstrap = BootstrapConfig(**data["bootstrap"])
        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])
        if "combinatorial" in data:
            config.combinatorial = CombinatorialConfig(**data["combinatorial"])
        if "cross_epistle" in data:
            config.cross_epistle = CrossEpistleConfig(**data["cross_epistle"])
        if "fractal" in data:
            config.fractal = FractalConfig(**data["fractal"])
        if "permutation" in data:
            config.permutation = PermutationConfig(**data["permutation"])
        if "vae" in data:
            config.vae = VAEConfig(**data["vae"])
        if "bayesian" in data:
            config.bayesian = BayesianConfig(**data["bayesian"])
        if "analysis" in data:
            config.analysis = AnalysisConfig(**data["analysis"])
        if "phases" in data:
            config.phases = data["phases"]

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import dataclasses
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = dataclasses.asdict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
