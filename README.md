# Pauline Neural Study

A novel computational approach to analyzing the Pauline corpus using bootstrap resampling, combinatorial recombination, and neural word embeddings — operating exclusively on Paul's own words.

## The Problem

Paul's epistles contain contested theological terms (e.g., *justification*, *faith*, *law*) whose precise meaning is debated across centuries of scholarship. Neural networks and word embeddings can reveal hidden semantic relationships, but Paul's corpus is small (~35,000 words across 7 undisputed epistles) — far below typical training requirements.

**The critical constraint:** We cannot supplement Paul's corpus with other authors' texts. Unlike coding or classification tasks where output can be validated against ground truth, theological interpretation has no objective validation metric. Any external contamination introduces undetectable bias into Paul's specific semantic meanings.

## The Solution

This study treats Paul's corpus as a **closed statistical universe** and applies mathematically rigorous expansion techniques that use *only Paul's words*:

1. **Bootstrap Resampling** — Inspired by the Central Limit Theorem: resample Paul's text with replacement to create thousands of valid "alternate Pauline corpora," then train embeddings on each. Stable word relationships across samples reveal Paul's authentic semantic structure.

2. **Combinatorial Recombination** — Extract Paul's syntactic templates and vocabulary-by-grammatical-slot mappings, then generate new sentences using only Paul's words in Paul's structures.

3. **Cross-Epistle Transfer** — Treat each epistle as an independent sample from Paul's mind. Leave-one-out analysis reveals which semantic relationships are universal vs. context-specific.

4. **Fractal Analysis** — Test whether Paul's language exhibits self-similarity at multiple scales (sentence, paragraph, epistle).

5. **Permutation Language Modeling** — Generate valid permutations of Paul's argument units for factorial training data expansion.

6. **Variational Autoencoder** — Learn a latent representation of "Pauline-ness" constrained to Paul's vocabulary.

7. **Hierarchical Bayesian Modeling** — Discover Paul's latent theological structure: core theology → epistle-specific applications → word choices.

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd pauline-neural-study

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Quick test run (reduced samples)
PYTHONPATH=src python -m pauline.pipeline --quick

# Full run
PYTHONPATH=src python -m pauline.pipeline --config configs/default.yaml
```

## GCP Deployment

For full-scale analysis on Google Cloud Platform:

```bash
# On the GCP VM:
git clone <repo-url> ~/pauline-neural-study
cd ~/pauline-neural-study
bash gcp/setup_vm.sh      # Install everything
bash gcp/run_analysis.sh   # Run the pipeline
```

See [gcp/setup_vm.sh](gcp/setup_vm.sh) for recommended VM specs.

## Project Structure

```
pauline-neural-study/
├── configs/
│   └── default.yaml              # Pipeline configuration
├── gcp/
│   ├── setup_vm.sh               # GCP VM setup script
│   └── run_analysis.sh           # GCP execution script
├── paper/
│   └── study_design.md           # Research paper / study design
├── src/pauline/
│   ├── pipeline.py               # Main pipeline orchestrator
│   ├── config.py                 # Configuration dataclasses
│   ├── corpus/
│   │   ├── loader.py             # Corpus loading and parsing
│   │   └── fetch.py              # API-based corpus fetching
│   ├── bootstrap/
│   │   └── sampler.py            # Multi-level bootstrap resampling
│   ├── embeddings/
│   │   └── trainer.py            # Word2Vec training + stability
│   ├── combinatorial/
│   │   └── recombiner.py         # Syntactic recombination
│   ├── cross_epistle/
│   │   └── analyzer.py           # Cross-epistle transfer analysis
│   ├── fractal/
│   │   └── analyzer.py           # Fractal self-similarity
│   ├── permutation/
│   │   └── plm.py                # Permutation language modeling
│   ├── vae/
│   │   └── model.py              # Variational autoencoder
│   ├── bayesian/
│   │   └── model.py              # Hierarchical Bayesian topics
│   └── analysis/
│       ├── semantic.py           # Semantic analysis tools
│       └── visualization.py      # Plotting and visualization
├── data/                         # Corpus data (fetched at runtime)
├── output/                       # Analysis results
└── tests/
    └── test_pipeline.py          # Unit tests
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Corpus**: Which epistles to include, translation, data source
- **Bootstrap**: Number of samples, sampling levels
- **Embeddings**: Dimensionality, window size, training epochs
- **Analysis**: Target theological terms, output options

## Running Specific Phases

```bash
# Only fetch corpus and run bootstrap
PYTHONPATH=src python -m pauline.pipeline --phases corpus bootstrap

# Only embeddings and analysis (requires prior bootstrap run)
PYTHONPATH=src python -m pauline.pipeline --phases embeddings analysis

# Include GPU-intensive VAE phase
PYTHONPATH=src python -m pauline.pipeline --phases corpus bootstrap embeddings vae analysis
```

## Research Paper

See [paper/study_design.md](paper/study_design.md) for a detailed explanation of the study design, theoretical foundations, and methodology.

## Requirements

- Python 3.10+
- ~16 GB RAM recommended for full pipeline
- GPU optional (only needed for VAE phase)
- Network access for initial corpus download (cached after first run)

## License

MIT
