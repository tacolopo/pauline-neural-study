"""
Pauline Variational Autoencoder (VAE)
=====================================

Learns a latent representation of "Pauline-ness" by training a
variational autoencoder on Paul's sentences. The encoder compresses
Pauline text into a continuous latent space; the decoder reconstructs
text constrained to Paul's vocabulary.

Theoretical Foundation:
-----------------------
A VAE learns a smooth, continuous latent space where nearby points
correspond to semantically similar texts. For the Pauline corpus,
this latent space captures the underlying dimensions along which
Paul's writing varies: theological emphasis, rhetorical register,
argumentative mode, emotional tone, etc.

Key properties:
    1. **Latent "Pauline-ness"**: Every point in the latent space
       decodes to text composed exclusively of Paul's vocabulary and
       syntactic patterns. The latent dimensions capture axes of
       variation WITHIN Paul's writing, not between Paul and others.

    2. **Interpolation**: Moving smoothly through latent space traces
       a path between different Pauline styles/topics. For example,
       interpolating between an encoded Romans passage and a Galatians
       passage reveals intermediate theological positions.

    3. **Generation**: Sampling from the prior p(z) generates new
       sentences that are statistically "Pauline" — they follow the
       distributional patterns of Paul's vocabulary and syntax.

    4. **Anomaly detection**: Text that is un-Pauline will have high
       reconstruction error and low likelihood under the model. This
       can be used to quantify stylistic deviation.

Architecture:
    - Input: Bag-of-words (BoW) representation over Paul's vocabulary
    - Encoder: Linear -> ReLU -> Linear -> (mu, log_var)
    - Latent: z ~ N(mu, sigma^2), reparameterization trick
    - Decoder: Linear -> ReLU -> Linear -> Softmax over vocabulary
    - Loss: Reconstruction (cross-entropy) + KL divergence

This is deliberately simple — a BoW-VAE rather than a sequential model —
because with a small corpus, model complexity must be kept low to avoid
overfitting. The BoW representation also aligns with the project's
distributional hypothesis: what matters is WHICH words Paul uses
together, not their exact order (order is handled by the PLM module).

Reference:
    - Kingma, D.P. & Welling, M. "Auto-Encoding Variational Bayes" (2014)
    - Bowman, S.R. et al. "Generating Sentences from a Continuous
      Space" (2016)
    - Miao, Y. et al. "Neural Variational Inference for Text
      Processing" (2016) — BoW-VAE for topic modeling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..corpus.loader import PaulineCorpus, word_tokenize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VAETrainingResult:
    """Results from VAE training.

    Attributes:
        train_losses: Per-epoch total loss values.
        recon_losses: Per-epoch reconstruction loss values.
        kl_losses: Per-epoch KL divergence values.
        final_loss: Final epoch total loss.
        vocabulary: The vocabulary used by the model.
        vocab_size: Size of the vocabulary.
        latent_dim: Dimensionality of the latent space.
    """
    train_losses: list[float] = field(default_factory=list)
    recon_losses: list[float] = field(default_factory=list)
    kl_losses: list[float] = field(default_factory=list)
    final_loss: float = 0.0
    vocabulary: list[str] = field(default_factory=list)
    vocab_size: int = 0
    latent_dim: int = 0


@dataclass
class LatentRepresentation:
    """Latent space encoding of a text.

    Attributes:
        text: Original input text.
        mu: Mean of the approximate posterior q(z|x).
        log_var: Log-variance of q(z|x).
        z: Sampled latent vector.
        reconstruction_error: Reconstruction loss for this input.
    """
    text: str
    mu: NDArray
    log_var: NDArray
    z: NDArray
    reconstruction_error: float


@dataclass
class GenerationResult:
    """Result from generating text via the decoder.

    Attributes:
        latent_vector: The z vector used for generation.
        word_probabilities: Probability distribution over vocabulary.
        top_words: Top-k most probable words with their probabilities.
        source_description: How the latent vector was obtained
            (e.g., "sampled from prior", "interpolation 0.5").
    """
    latent_vector: NDArray
    word_probabilities: NDArray
    top_words: list[tuple[str, float]]
    source_description: str


@dataclass
class AnomalyResult:
    """Result from anomaly scoring of a text.

    Attributes:
        text: The text being scored.
        reconstruction_error: How poorly the VAE reconstructs the input.
        kl_divergence: KL divergence of the posterior from the prior.
        anomaly_score: Combined score (higher = more anomalous = less Pauline).
        percentile: Where this score falls relative to the training corpus.
    """
    text: str
    reconstruction_error: float
    kl_divergence: float
    anomaly_score: float
    percentile: float


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PaulineBoWDataset(Dataset):
    """
    Bag-of-words dataset for Pauline sentences.

    Each sample is a normalized term-frequency vector over Paul's
    vocabulary.
    """

    def __init__(
        self,
        sentences: list[str],
        vocabulary: list[str],
    ):
        self.vocabulary = vocabulary
        self.word_to_idx = {w: i for i, w in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)

        # Pre-compute BoW vectors
        self.bow_vectors: list[torch.Tensor] = []
        for sent in sentences:
            vec = self._sentence_to_bow(sent)
            self.bow_vectors.append(vec)

    def _sentence_to_bow(self, sentence: str) -> torch.Tensor:
        """Convert a sentence to a normalized BoW vector."""
        tokens = word_tokenize(sentence)
        vec = torch.zeros(self.vocab_size, dtype=torch.float32)
        for t in tokens:
            if t in self.word_to_idx:
                vec[self.word_to_idx[t]] += 1.0
        # Normalize to sum to 1 (probability distribution)
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

    def __len__(self) -> int:
        return len(self.bow_vectors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.bow_vectors[idx]


# ---------------------------------------------------------------------------
# VAE architecture (PyTorch)
# ---------------------------------------------------------------------------

class _VAEModule(nn.Module):
    """
    Variational Autoencoder for bag-of-words text.

    Architecture:
        Encoder: input_dim -> hidden_dim -> (mu, log_var)  [latent_dim each]
        Decoder: latent_dim -> hidden_dim -> output_dim (= input_dim)

    The decoder outputs a probability distribution over the vocabulary
    (via softmax), representing the expected word frequencies for a
    Pauline text at the given point in latent space.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
            # No activation — log_softmax applied in loss
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters (mu, log_var)."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.

        Allows gradients to flow through the sampling operation.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to vocabulary distribution (logits)."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode, sample, decode.

        Returns:
            (reconstructed_logits, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def vae_loss(
    recon_logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss = Reconstruction loss + KL divergence.

    Reconstruction: cross-entropy between the decoded distribution
    and the input BoW distribution.

    KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I).
    Closed-form for two Gaussians:
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Args:
        recon_logits: Decoder output (pre-softmax).
        target: Input BoW vector (normalized).
        mu: Encoder mean.
        log_var: Encoder log-variance.
        kl_weight: Scaling factor for KL term (beta-VAE).

    Returns:
        (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction: treat as multi-label soft cross-entropy
    recon_loss = -torch.sum(
        target * F.log_softmax(recon_logits, dim=-1), dim=-1
    ).mean()

    # KL divergence
    kl_div = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp(), dim=-1
    ).mean()

    total = recon_loss + kl_weight * kl_div
    return total, recon_loss, kl_div


# ---------------------------------------------------------------------------
# Main PaulineVAE class
# ---------------------------------------------------------------------------

class PaulineVAE:
    """
    Variational autoencoder for learning latent Pauline text structure.

    Trains a BoW-VAE on Paul's sentences to learn a continuous latent
    space where every point decodes to a Pauline vocabulary distribution.
    Supports encoding, generation, interpolation, and anomaly detection.

    Usage::

        from pauline.corpus.loader import PaulineCorpus
        from pauline.vae.model import PaulineVAE

        corpus = PaulineCorpus.from_json("pauline_corpus.json")
        vae = PaulineVAE(corpus, latent_dim=32)
        result = vae.train(n_epochs=100)

        # Encode a passage
        rep = vae.encode_text("for by grace you have been saved through faith")
        print(f"Latent vector: {rep.z[:5]}...")

        # Generate Pauline vocabulary distribution
        gen = vae.generate()
        print(f"Top words: {gen.top_words[:10]}")

        # Anomaly detection
        score = vae.anomaly_score("totally non-pauline modern text here")
        print(f"Anomaly score: {score.anomaly_score:.3f}")
    """

    def __init__(
        self,
        corpus: PaulineCorpus,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        min_word_freq: int = 2,
        dropout: float = 0.2,
        device: Optional[str] = None,
    ):
        self.corpus = corpus
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.min_word_freq = min_word_freq
        self.dropout = dropout

        # Device selection
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Build vocabulary
        self._build_vocabulary()

        # Initialize model
        self.model = _VAEModule(
            vocab_size=self.vocab_size,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        ).to(self.device)

        # Will store training corpus anomaly scores for percentile computation
        self._training_scores: Optional[NDArray] = None

    def _build_vocabulary(self) -> None:
        """Build the Pauline vocabulary for BoW representation."""
        word_freq = self.corpus.word_frequency()
        self.vocabulary = sorted(
            w for w, freq in word_freq.items()
            if freq >= self.min_word_freq and w.isalpha()
        )
        self.word_to_idx = {w: i for i, w in enumerate(self.vocabulary)}
        self.vocab_size = len(self.vocabulary)

        logger.info(
            f"VAE vocabulary: {self.vocab_size} words "
            f"(min freq >= {self.min_word_freq})"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        kl_weight: float = 1.0,
        kl_annealing: bool = True,
        kl_annealing_epochs: int = 20,
    ) -> VAETrainingResult:
        """
        Train the VAE on the Pauline corpus.

        Uses KL annealing (gradually increasing the KL weight from 0
        to ``kl_weight`` over the first ``kl_annealing_epochs``) to
        prevent posterior collapse, a common issue in text VAEs where
        the model ignores the latent variable.

        Args:
            n_epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Adam optimizer learning rate.
            kl_weight: Maximum KL divergence weight (beta in beta-VAE).
            kl_annealing: Whether to anneal the KL weight.
            kl_annealing_epochs: Epochs over which to linearly increase
                the KL weight from 0 to ``kl_weight``.

        Returns:
            VAETrainingResult with loss curves and metadata.
        """
        # Prepare dataset
        all_sentences = [s for _, s in self.corpus.all_sentences]
        dataset = PaulineBoWDataset(all_sentences, self.vocabulary)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        result = VAETrainingResult(
            vocabulary=self.vocabulary,
            vocab_size=self.vocab_size,
            latent_dim=self.latent_dim,
        )

        logger.info(
            f"Training VAE: {len(dataset)} sentences, "
            f"{self.vocab_size} vocab, {self.latent_dim}-dim latent, "
            f"device={self.device}"
        )

        self.model.train()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0

            # KL annealing: linear ramp
            if kl_annealing and epoch < kl_annealing_epochs:
                current_kl_weight = kl_weight * (epoch / kl_annealing_epochs)
            else:
                current_kl_weight = kl_weight

            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                recon, mu, log_var = self.model(batch)
                total, recon_loss, kl_div = vae_loss(
                    recon, batch, mu, log_var,
                    kl_weight=current_kl_weight,
                )

                total.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )
                optimizer.step()

                epoch_loss += total.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_div.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_recon = epoch_recon / max(n_batches, 1)
            avg_kl = epoch_kl / max(n_batches, 1)

            result.train_losses.append(avg_loss)
            result.recon_losses.append(avg_recon)
            result.kl_losses.append(avg_kl)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"loss={avg_loss:.4f} "
                    f"(recon={avg_recon:.4f}, kl={avg_kl:.4f}, "
                    f"kl_w={current_kl_weight:.3f})"
                )

        result.final_loss = result.train_losses[-1] if result.train_losses else 0.0

        # Compute training corpus anomaly scores for percentile calibration
        self._calibrate_anomaly_scores(all_sentences)

        logger.info(f"Training complete. Final loss: {result.final_loss:.4f}")
        return result

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> LatentRepresentation:
        """
        Encode a text passage to its latent representation.

        Args:
            text: Input text (will be tokenized and converted to BoW).

        Returns:
            LatentRepresentation with mu, log_var, z, and reconstruction error.
        """
        self.model.eval()

        bow_dataset = PaulineBoWDataset([text], self.vocabulary)
        bow_vec = bow_dataset[0].unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, log_var = self.model.encode(bow_vec)
            z = self.model.reparameterize(mu, log_var)
            recon = self.model.decode(z)
            _, recon_loss, _ = vae_loss(recon, bow_vec, mu, log_var, kl_weight=0)

        return LatentRepresentation(
            text=text,
            mu=mu.cpu().numpy().flatten(),
            log_var=log_var.cpu().numpy().flatten(),
            z=z.cpu().numpy().flatten(),
            reconstruction_error=float(recon_loss.item()),
        )

    def encode_corpus(self) -> list[LatentRepresentation]:
        """Encode all sentences in the corpus to latent vectors."""
        representations = []
        for _, sent in self.corpus.all_sentences:
            rep = self.encode_text(sent)
            representations.append(rep)
        return representations

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        z: Optional[NDArray] = None,
        top_k: int = 20,
        description: str = "sampled from prior",
    ) -> GenerationResult:
        """
        Generate a Pauline vocabulary distribution from a latent vector.

        Args:
            z: Latent vector. If None, sample from the prior N(0, I).
            top_k: Number of top words to return.
            description: Description of how z was obtained.

        Returns:
            GenerationResult with word probabilities and top words.
        """
        self.model.eval()

        if z is None:
            z_np = np.random.randn(self.latent_dim).astype(np.float32)
            description = "sampled from prior N(0, I)"
        else:
            z_np = z.astype(np.float32)

        z_tensor = torch.from_numpy(z_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model.decode(z_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()

        # Get top-k words
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_words = [
            (self.vocabulary[i], float(probs[i]))
            for i in top_indices
        ]

        return GenerationResult(
            latent_vector=z_np,
            word_probabilities=probs,
            top_words=top_words,
            source_description=description,
        )

    def interpolate(
        self,
        text_a: str,
        text_b: str,
        n_steps: int = 5,
        top_k: int = 15,
    ) -> list[GenerationResult]:
        """
        Interpolate between two texts in latent space.

        Encodes both texts, then generates vocabulary distributions at
        evenly spaced points along the line connecting their latent
        representations. This reveals intermediate "Pauline states"
        between two passages.

        Args:
            text_a: Starting text.
            text_b: Ending text.
            n_steps: Number of interpolation steps (including endpoints).
            top_k: Number of top words per step.

        Returns:
            List of GenerationResults along the interpolation path.
        """
        rep_a = self.encode_text(text_a)
        rep_b = self.encode_text(text_b)

        results = []
        for i in range(n_steps):
            alpha = i / max(n_steps - 1, 1)
            z = (1 - alpha) * rep_a.mu + alpha * rep_b.mu
            result = self.generate(
                z=z,
                top_k=top_k,
                description=f"interpolation alpha={alpha:.2f}",
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def anomaly_score(self, text: str) -> AnomalyResult:
        """
        Score how "un-Pauline" a text is.

        Combines reconstruction error (how well the VAE can reconstruct
        the text using Pauline patterns) with KL divergence (how far
        the text's latent representation is from the prior).

        High scores indicate the text deviates from Pauline patterns.

        Args:
            text: Input text to score.

        Returns:
            AnomalyResult with scores and percentile.
        """
        self.model.eval()

        bow_dataset = PaulineBoWDataset([text], self.vocabulary)
        bow_vec = bow_dataset[0].unsqueeze(0).to(self.device)

        with torch.no_grad():
            recon, mu, log_var = self.model(bow_vec)
            _, recon_loss, kl_div = vae_loss(
                recon, bow_vec, mu, log_var, kl_weight=1.0
            )

        recon_val = float(recon_loss.item())
        kl_val = float(kl_div.item())
        combined = recon_val + kl_val

        # Compute percentile relative to training corpus
        percentile = 0.0
        if self._training_scores is not None and len(self._training_scores) > 0:
            percentile = float(
                np.mean(self._training_scores <= combined) * 100.0
            )

        return AnomalyResult(
            text=text,
            reconstruction_error=recon_val,
            kl_divergence=kl_val,
            anomaly_score=combined,
            percentile=percentile,
        )

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and vocabulary to disk."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "vocabulary": self.vocabulary,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
            "training_scores": self._training_scores,
        }, path)
        logger.info(f"VAE model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights and vocabulary from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.vocabulary = checkpoint["vocabulary"]
        self.vocab_size = checkpoint["vocab_size"]
        self.latent_dim = checkpoint["latent_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocabulary)}
        self._training_scores = checkpoint.get("training_scores")

        self.model = _VAEModule(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"VAE model loaded from {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calibrate_anomaly_scores(self, sentences: list[str]) -> None:
        """Compute anomaly scores for all training sentences to
        establish the baseline distribution for percentile computation."""
        self.model.eval()

        scores = []
        dataset = PaulineBoWDataset(sentences, self.vocabulary)

        with torch.no_grad():
            for i in range(len(dataset)):
                bow_vec = dataset[i].unsqueeze(0).to(self.device)
                recon, mu, log_var = self.model(bow_vec)
                _, recon_loss, kl_div = vae_loss(
                    recon, bow_vec, mu, log_var, kl_weight=1.0
                )
                scores.append(recon_loss.item() + kl_div.item())

        self._training_scores = np.array(scores)
        logger.info(
            f"Calibrated anomaly baseline: "
            f"mean={np.mean(self._training_scores):.4f}, "
            f"std={np.std(self._training_scores):.4f}"
        )
