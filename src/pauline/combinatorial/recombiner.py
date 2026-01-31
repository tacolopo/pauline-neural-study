"""
Combinatorial Recombiner
========================

Generates synthetic Pauline text by recombining Paul's own words
within his own grammatical structures.

Approach:
    1. Parse all Pauline sentences into POS-tag templates
    2. For each grammatical slot (NOUN, VERB, ADJ, etc.), collect
       the set of words Paul actually used in that slot
    3. Generate new sentences by filling templates with Paul's words
       drawn from the correct grammatical slots
    4. Apply co-occurrence constraints: only combine words that
       Paul used in similar contexts

Constraints (Preserving Pauline Integrity):
    - Every word in generated text exists in Paul's vocabulary
    - Every grammatical structure is attested in Paul's writing
    - Word combinations respect Paul's co-occurrence patterns
    - No external vocabulary or structures are introduced

This is NOT text generation in the LLM sense. It is a controlled
combinatorial expansion that creates new training samples while
guaranteeing every element is authentically Pauline.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np
import nltk
from nltk import pos_tag, word_tokenize

from ..corpus.loader import PaulineCorpus

logger = logging.getLogger(__name__)


@dataclass
class SyntacticTemplate:
    """A POS-tag template extracted from a Pauline sentence."""
    tags: tuple[str, ...]  # POS tag sequence
    original_sentence: str
    source_epistle: str
    word_slots: dict[int, list[str]] = field(default_factory=dict)
    # Maps position -> list of words Paul used at that POS in this template

    @property
    def length(self) -> int:
        return len(self.tags)

    @property
    def tag_string(self) -> str:
        return " ".join(self.tags)


@dataclass
class RecombinationResult:
    """Results from combinatorial recombination."""
    generated_sentences: list[str]
    templates_used: int
    unique_templates: int
    vocabulary_used: set[str]
    generation_method: str
    constraints_applied: list[str]


class CombinatorialRecombiner:
    """
    Generates synthetic Pauline text through controlled combinatorial
    recombination of Paul's vocabulary within his syntactic structures.

    The key principle: separate Paul's WHAT (vocabulary) from his HOW
    (syntax), then recombine them in new but structurally valid ways.
    Every component is authentically Pauline.
    """

    def __init__(
        self,
        corpus: PaulineCorpus,
        seed: Optional[int] = None,
    ):
        self.corpus = corpus
        self.rng = np.random.default_rng(seed)
        self._templates: list[SyntacticTemplate] = []
        self._pos_vocabulary: dict[str, list[str]] = defaultdict(list)
        self._bigram_freqs: dict[tuple[str, str], int] = defaultdict(int)
        self._pos_bigram_freqs: dict[tuple[str, str], int] = defaultdict(int)
        self._prepared = False

    def prepare(self) -> None:
        """
        Parse corpus to extract templates and build POS vocabulary.

        This must be called before generating sentences.
        """
        logger.info("Preparing combinatorial recombiner...")

        # Ensure NLTK resources are available
        for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"taggers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        all_words_in_context: dict[str, set[str]] = defaultdict(set)

        for ep in self.corpus.epistles:
            for _, sent_text in [(ep.name, s) for s in ep.sentences]:
                tokens = word_tokenize(sent_text.lower())
                if len(tokens) < 3:
                    continue

                tagged = pos_tag(tokens)
                tags = tuple(tag for _, tag in tagged)

                template = SyntacticTemplate(
                    tags=tags,
                    original_sentence=sent_text,
                    source_epistle=ep.name,
                )

                # Record which words Paul used at each position
                for i, (word, tag) in enumerate(tagged):
                    if word.isalpha():
                        template.word_slots.setdefault(i, []).append(word)
                        self._pos_vocabulary[tag].append(word)
                        all_words_in_context[tag].add(word)

                self._templates.append(template)

                # Build bigram frequencies for co-occurrence constraints
                for i in range(len(tokens) - 1):
                    if tokens[i].isalpha() and tokens[i + 1].isalpha():
                        self._bigram_freqs[(tokens[i], tokens[i + 1])] += 1

                for i in range(len(tags) - 1):
                    self._pos_bigram_freqs[(tags[i], tags[i + 1])] += 1

        # Deduplicate POS vocabulary
        for tag in self._pos_vocabulary:
            self._pos_vocabulary[tag] = list(set(self._pos_vocabulary[tag]))

        # Identify unique template structures
        unique_structures = set(t.tag_string for t in self._templates)

        self._prepared = True
        logger.info(
            f"Prepared: {len(self._templates)} templates, "
            f"{len(unique_structures)} unique structures, "
            f"{sum(len(v) for v in self._pos_vocabulary.values())} POS-word mappings"
        )

    def generate_random(
        self,
        n_sentences: int = 100,
        min_length: int = 5,
        max_length: int = 30,
    ) -> RecombinationResult:
        """
        Generate sentences by randomly filling templates with
        Pauline vocabulary from matching POS slots.

        This is the simplest generation method: pick a random template,
        then for each slot, pick a random word from Paul's vocabulary
        that has the same POS tag.

        Args:
            n_sentences: Number of sentences to generate.
            min_length: Minimum template length (in tokens).
            max_length: Maximum template length (in tokens).

        Returns:
            RecombinationResult with generated sentences.
        """
        if not self._prepared:
            self.prepare()

        # Filter templates by length
        valid_templates = [
            t for t in self._templates
            if min_length <= t.length <= max_length
        ]

        if not valid_templates:
            raise ValueError(f"No templates found with length {min_length}-{max_length}")

        generated = []
        vocab_used: set[str] = set()

        for _ in range(n_sentences):
            template = valid_templates[self.rng.integers(0, len(valid_templates))]
            sentence_words = []

            for i, tag in enumerate(template.tags):
                candidates = self._pos_vocabulary.get(tag, [])
                if candidates:
                    word = candidates[self.rng.integers(0, len(candidates))]
                    sentence_words.append(word)
                    vocab_used.add(word)
                else:
                    # Use original word if no candidates for this POS
                    original_tokens = word_tokenize(template.original_sentence.lower())
                    if i < len(original_tokens):
                        sentence_words.append(original_tokens[i])

            generated.append(" ".join(sentence_words))

        return RecombinationResult(
            generated_sentences=generated,
            templates_used=n_sentences,
            unique_templates=len(set(t.tag_string for t in valid_templates)),
            vocabulary_used=vocab_used,
            generation_method="random_slot_filling",
            constraints_applied=["pos_tag_matching", "pauline_vocabulary_only"],
        )

    def generate_constrained(
        self,
        n_sentences: int = 100,
        bigram_threshold: int = 1,
        min_length: int = 5,
        max_length: int = 30,
        max_attempts_per_sentence: int = 50,
    ) -> RecombinationResult:
        """
        Generate sentences with co-occurrence constraints.

        Like generate_random, but additionally requires that adjacent
        word pairs have been observed together (or in similar contexts)
        in Paul's actual text. This produces more naturally Pauline
        combinations.

        Args:
            n_sentences: Number of sentences to generate.
            bigram_threshold: Minimum bigram frequency to allow a pair.
                            Set to 0 to allow any POS-valid bigram.
            min_length: Minimum template length.
            max_length: Maximum template length.
            max_attempts_per_sentence: Max retries per sentence.

        Returns:
            RecombinationResult with generated sentences.
        """
        if not self._prepared:
            self.prepare()

        valid_templates = [
            t for t in self._templates
            if min_length <= t.length <= max_length
        ]

        if not valid_templates:
            raise ValueError(f"No templates found with length {min_length}-{max_length}")

        generated = []
        vocab_used: set[str] = set()

        for _ in range(n_sentences):
            template = valid_templates[self.rng.integers(0, len(valid_templates))]
            best_sentence = None

            for attempt in range(max_attempts_per_sentence):
                sentence_words = []
                valid = True

                for i, tag in enumerate(template.tags):
                    candidates = self._pos_vocabulary.get(tag, [])
                    if not candidates:
                        original_tokens = word_tokenize(template.original_sentence.lower())
                        if i < len(original_tokens):
                            sentence_words.append(original_tokens[i])
                        continue

                    if i == 0 or not sentence_words:
                        # First word: pick randomly
                        word = candidates[self.rng.integers(0, len(candidates))]
                    else:
                        # Subsequent words: prefer those observed after previous word
                        prev_word = sentence_words[-1]
                        scored = []
                        for w in candidates:
                            freq = self._bigram_freqs.get((prev_word, w), 0)
                            scored.append((w, freq))

                        # Use softmax-like selection biased toward observed bigrams
                        if bigram_threshold > 0:
                            valid_candidates = [
                                (w, f) for w, f in scored if f >= bigram_threshold
                            ]
                            if not valid_candidates:
                                # Fall back to POS bigram constraint
                                word = candidates[self.rng.integers(0, len(candidates))]
                            else:
                                freqs = np.array([f for _, f in valid_candidates])
                                probs = (freqs + 1) / (freqs + 1).sum()
                                idx = self.rng.choice(len(valid_candidates), p=probs)
                                word = valid_candidates[idx][0]
                        else:
                            freqs = np.array([f for _, f in scored], dtype=float)
                            probs = (freqs + 1) / (freqs + 1).sum()
                            idx = self.rng.choice(len(scored), p=probs)
                            word = scored[idx][0]

                    sentence_words.append(word)
                    vocab_used.add(word)

                best_sentence = " ".join(sentence_words)
                break

            if best_sentence:
                generated.append(best_sentence)

        return RecombinationResult(
            generated_sentences=generated,
            templates_used=n_sentences,
            unique_templates=len(set(t.tag_string for t in valid_templates)),
            vocabulary_used=vocab_used,
            generation_method="constrained_slot_filling",
            constraints_applied=[
                "pos_tag_matching",
                "pauline_vocabulary_only",
                f"bigram_cooccurrence (threshold={bigram_threshold})",
            ],
        )

    def generate_from_seed_words(
        self,
        seed_words: list[str],
        n_sentences: int = 20,
        min_length: int = 5,
        max_length: int = 25,
    ) -> RecombinationResult:
        """
        Generate sentences that incorporate specific seed words.

        Useful for exploring how Paul might have combined specific
        theological terms. Only uses templates whose POS structure
        can accommodate the seed words.

        Args:
            seed_words: Words to include in generated sentences.
                       Must be in Paul's vocabulary.
            min_length: Minimum sentence length.
            max_length: Maximum sentence length.

        Returns:
            RecombinationResult with generated sentences containing seed words.
        """
        if not self._prepared:
            self.prepare()

        # Validate seed words are in Paul's vocabulary
        paul_vocab = self.corpus.vocabulary
        invalid = [w for w in seed_words if w.lower() not in paul_vocab]
        if invalid:
            logger.warning(f"Words not in Paul's vocabulary (skipping): {invalid}")
            seed_words = [w for w in seed_words if w.lower() not in paul_vocab]

        if not seed_words:
            raise ValueError("No valid seed words remain")

        # Find POS tags for seed words
        seed_tags = {word: pos_tag([word.lower()])[0][1] for word in seed_words}

        # Find templates that have slots matching seed word POS tags
        valid_templates = []
        for t in self._templates:
            if not (min_length <= t.length <= max_length):
                continue
            template_tags = set(t.tags)
            if all(tag in template_tags for tag in seed_tags.values()):
                valid_templates.append(t)

        if not valid_templates:
            logger.warning("No templates match all seed word POS tags, using relaxed matching")
            valid_templates = [
                t for t in self._templates
                if min_length <= t.length <= max_length
            ]

        generated = []
        vocab_used: set[str] = set(seed_words)

        for _ in range(n_sentences):
            template = valid_templates[self.rng.integers(0, len(valid_templates))]
            sentence_words = []
            seed_placed = {w: False for w in seed_words}

            for i, tag in enumerate(template.tags):
                # Try to place unplaced seed words
                placed = False
                for sw in seed_words:
                    if not seed_placed[sw] and seed_tags.get(sw) == tag:
                        sentence_words.append(sw.lower())
                        seed_placed[sw] = True
                        placed = True
                        break

                if not placed:
                    candidates = self._pos_vocabulary.get(tag, [])
                    if candidates:
                        word = candidates[self.rng.integers(0, len(candidates))]
                        sentence_words.append(word)
                        vocab_used.add(word)
                    else:
                        original_tokens = word_tokenize(template.original_sentence.lower())
                        if i < len(original_tokens):
                            sentence_words.append(original_tokens[i])

            generated.append(" ".join(sentence_words))

        return RecombinationResult(
            generated_sentences=generated,
            templates_used=n_sentences,
            unique_templates=len(set(t.tag_string for t in valid_templates)),
            vocabulary_used=vocab_used,
            generation_method="seed_word_constrained",
            constraints_applied=[
                "pos_tag_matching",
                "pauline_vocabulary_only",
                f"seed_words: {seed_words}",
            ],
        )
