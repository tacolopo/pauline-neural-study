"""
Pauline Corpus Loader
=====================

Loads, parses, and organizes the Pauline epistles at multiple granularity
levels: epistle, chapter, pericope (argument unit), and sentence.

Supports both Greek (Koine) and English text. The statistical methods in
this study are language-agnostic — they operate on token distributions
regardless of source language. Greek is preferred for scholarly precision;
English enables rapid prototyping and validation.

Greek Tokenization:
    Koine Greek uses distinct punctuation conventions:
        . (period)         → sentence end
        · (ano teleia)     → clause separator (like English semicolon/colon)
        ; (erotimatiko)    → question mark
        , (comma)          → clause separator
    Sentence boundaries are detected at . and ; marks.
    Word tokenization splits on whitespace and strips punctuation.

The 7 undisputed Pauline epistles:
    - Romans
    - 1 Corinthians
    - 2 Corinthians
    - Galatians
    - Philippians
    - 1 Thessalonians
    - Philemon

The 6 disputed epistles (included optionally):
    - Ephesians
    - Colossians
    - 2 Thessalonians
    - 1 Timothy
    - 2 Timothy
    - Titus

Additionally, Hebrews is included in the corpus data (traditionally
attributed to Paul, though widely disputed).
"""

from __future__ import annotations

import json
import re
import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Epistle metadata: book name -> (abbreviation, undisputed?)
PAULINE_EPISTLES = {
    "Romans": ("Rom", True),
    "1 Corinthians": ("1Cor", True),
    "2 Corinthians": ("2Cor", True),
    "Galatians": ("Gal", True),
    "Philippians": ("Phil", True),
    "1 Thessalonians": ("1Thess", True),
    "Philemon": ("Phlm", True),
    "Ephesians": ("Eph", False),
    "Colossians": ("Col", False),
    "2 Thessalonians": ("2Thess", False),
    "1 Timothy": ("1Tim", False),
    "2 Timothy": ("2Tim", False),
    "Titus": ("Tit", False),
    "Hebrews": ("Heb", False),
}

UNDISPUTED_EPISTLES = [name for name, (_, und) in PAULINE_EPISTLES.items() if und]
ALL_EPISTLES = list(PAULINE_EPISTLES.keys())


# ---------------------------------------------------------------------------
# Greek tokenization utilities
# ---------------------------------------------------------------------------

def _is_greek(text: str) -> bool:
    """Detect if text is primarily Greek by checking character scripts."""
    greek_chars = sum(1 for ch in text if unicodedata.category(ch).startswith("L")
                      and ("\u0370" <= ch <= "\u03FF" or "\u1F00" <= ch <= "\u1FFF"))
    latin_chars = sum(1 for ch in text if unicodedata.category(ch).startswith("L")
                      and "A" <= ch <= "z")
    return greek_chars > latin_chars


def greek_sent_tokenize(text: str) -> list[str]:
    """
    Split Greek text into sentences.

    Sentence boundaries in Koine Greek:
        . (U+002E, period) — full stop
        ; (U+003B, semicolon) — Greek question mark (erotimatiko)

    The ano teleia (·, U+00B7 or U+0387) is a clause separator
    (similar to English semicolon) but NOT a sentence boundary.
    """
    # Split on sentence-ending punctuation followed by space or end-of-string.
    # Preserve the punctuation with the preceding sentence.
    parts = re.split(r'(?<=[.;])\s+', text.strip())
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


def greek_word_tokenize(text: str) -> list[str]:
    """
    Tokenize Greek text into words.

    Splits on whitespace and strips punctuation characters.
    Preserves Greek diacriticals and breathing marks.
    Lowercasing is not applied here since Greek has meaningful case
    (though the caller may choose to normalize).
    """
    # Split on whitespace
    raw_tokens = text.split()
    words = []
    for token in raw_tokens:
        # Strip leading/trailing punctuation (.,;·—()«»)
        cleaned = token.strip(".,;·\u0387—–-()«»\"'[]{}!?:''""")
        if cleaned:
            words.append(cleaned)
    return words


def normalize_greek(word: str) -> str:
    """
    Normalize a Greek word for comparison purposes.

    Applies Unicode NFC normalization and lowercases.
    Greek lowercasing handles final sigma (ς vs σ) correctly.
    """
    return unicodedata.normalize("NFC", word.lower())


def sent_tokenize(text: str) -> list[str]:
    """Language-aware sentence tokenizer. Uses Greek or NLTK English."""
    if _is_greek(text):
        return greek_sent_tokenize(text)
    else:
        import nltk
        return nltk.sent_tokenize(text)


def word_tokenize(text: str) -> list[str]:
    """Language-aware word tokenizer. Uses Greek or NLTK English."""
    if _is_greek(text):
        return [normalize_greek(w) for w in greek_word_tokenize(text)]
    else:
        import nltk
        return nltk.word_tokenize(text.lower())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Verse:
    """A single verse from a Pauline epistle."""
    book: str
    chapter: int
    verse: int
    text: str

    @property
    def reference(self) -> str:
        abbr = PAULINE_EPISTLES.get(self.book, (self.book,))[0]
        return f"{abbr} {self.chapter}:{self.verse}"


@dataclass
class Pericope:
    """
    A pericope (coherent argument unit) within a Pauline epistle.

    Pericopes are defined by scholarly consensus as self-contained
    theological argument units. They are the natural unit for
    permutation-based analysis, as they preserve theological coherence.
    """
    book: str
    label: str
    verses: list[Verse] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(v.text for v in self.verses)

    @property
    def sentences(self) -> list[str]:
        return sent_tokenize(self.text)

    @property
    def words(self) -> list[str]:
        return word_tokenize(self.text)


@dataclass
class Epistle:
    """A complete Pauline epistle organized by chapter and verse."""
    name: str
    verses: list[Verse] = field(default_factory=list)
    pericopes: list[Pericope] = field(default_factory=list)

    @property
    def abbreviation(self) -> str:
        return PAULINE_EPISTLES.get(self.name, (self.name,))[0]

    @property
    def is_undisputed(self) -> bool:
        return PAULINE_EPISTLES.get(self.name, (None, False))[1]

    @property
    def text(self) -> str:
        return " ".join(v.text for v in self.verses)

    @property
    def sentences(self) -> list[str]:
        return sent_tokenize(self.text)

    @property
    def words(self) -> list[str]:
        return word_tokenize(self.text)

    @property
    def chapters(self) -> dict[int, list[Verse]]:
        chaps: dict[int, list[Verse]] = {}
        for v in self.verses:
            chaps.setdefault(v.chapter, []).append(v)
        return chaps

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def vocabulary(self) -> set[str]:
        return set(self.words)


class PaulineCorpus:
    """
    The complete Pauline corpus, organized hierarchically.

    Provides access at multiple granularity levels:
        - Corpus level: all epistles combined
        - Epistle level: individual letters
        - Chapter level: chapters within epistles
        - Pericope level: theological argument units
        - Sentence level: individual sentences
        - Word level: individual tokens

    Each level is a valid unit for bootstrap resampling.
    """

    def __init__(
        self,
        epistles: list[Epistle],
        undisputed_only: bool = True,
    ):
        if undisputed_only:
            self.epistles = [e for e in epistles if e.is_undisputed]
        else:
            self.epistles = epistles

        self.undisputed_only = undisputed_only
        self._build_indices()

    def _build_indices(self) -> None:
        """Build lookup indices for efficient access."""
        self._by_name: dict[str, Epistle] = {}
        self._all_sentences: list[tuple[str, str]] = []  # (book, sentence)
        self._all_words: list[str] = []
        self._vocabulary: set[str] = set()

        for ep in self.epistles:
            self._by_name[ep.name] = ep
            for s in ep.sentences:
                self._all_sentences.append((ep.name, s))
            self._all_words.extend(ep.words)
            self._vocabulary.update(ep.vocabulary)

    def get_epistle(self, name: str) -> Optional[Epistle]:
        return self._by_name.get(name)

    @property
    def epistle_names(self) -> list[str]:
        return [e.name for e in self.epistles]

    @property
    def all_text(self) -> str:
        return " ".join(e.text for e in self.epistles)

    @property
    def all_sentences(self) -> list[tuple[str, str]]:
        """All sentences as (book_name, sentence_text) tuples."""
        return self._all_sentences

    @property
    def all_words(self) -> list[str]:
        return self._all_words

    @property
    def vocabulary(self) -> set[str]:
        return self._vocabulary

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    @property
    def total_words(self) -> int:
        return len(self._all_words)

    @property
    def total_sentences(self) -> int:
        return len(self._all_sentences)

    @property
    def is_greek(self) -> bool:
        """Check if corpus is in Greek."""
        if self.epistles:
            sample = self.epistles[0].text[:200]
            return _is_greek(sample)
        return False

    def sentences_by_epistle(self) -> dict[str, list[str]]:
        """Group sentences by their source epistle."""
        result: dict[str, list[str]] = {}
        for book, s in self._all_sentences:
            result.setdefault(book, []).append(s)
        return result

    def word_frequency(self) -> dict[str, int]:
        """Word frequency distribution across the entire corpus."""
        freq: dict[str, int] = {}
        for w in self._all_words:
            freq[w] = freq.get(w, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: -x[1]))

    def summary(self) -> dict:
        """Summary statistics of the corpus."""
        return {
            "epistles": len(self.epistles),
            "epistle_names": self.epistle_names,
            "total_words": self.total_words,
            "vocabulary_size": self.vocabulary_size,
            "total_sentences": self.total_sentences,
            "undisputed_only": self.undisputed_only,
            "is_greek": self.is_greek,
            "words_per_epistle": {
                e.name: e.word_count for e in self.epistles
            },
        }

    @classmethod
    def from_json(cls, path: str | Path, undisputed_only: bool = True) -> PaulineCorpus:
        """
        Load corpus from a JSON file.

        Expected format:
        {
            "epistles": [
                {
                    "name": "Romans",
                    "verses": [
                        {"chapter": 1, "verse": 1, "text": "..."},
                        ...
                    ]
                },
                ...
            ]
        }
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        epistles = []
        for ep_data in data["epistles"]:
            verses = [
                Verse(
                    book=ep_data["name"],
                    chapter=v["chapter"],
                    verse=v["verse"],
                    text=v["text"],
                )
                for v in ep_data["verses"]
            ]
            epistle = Epistle(name=ep_data["name"], verses=verses)

            # Load pericopes if present
            if "pericopes" in ep_data:
                for p_data in ep_data["pericopes"]:
                    start_ch, start_v = p_data["start_ref"]
                    end_ch, end_v = p_data["end_ref"]
                    peri_verses = [
                        v for v in verses
                        if (start_ch, start_v) <= (v.chapter, v.verse) <= (end_ch, end_v)
                    ]
                    epistle.pericopes.append(
                        Pericope(
                            book=ep_data["name"],
                            label=p_data.get("label", ""),
                            verses=peri_verses,
                        )
                    )

            epistles.append(epistle)

        return cls(epistles=epistles, undisputed_only=undisputed_only)

    @classmethod
    def from_text_files(cls, directory: str | Path, undisputed_only: bool = True) -> PaulineCorpus:
        """
        Load corpus from plain text files.

        Each file should be named after the epistle (e.g., 'Romans.txt').
        Supports both Greek and English text. Language is auto-detected
        and the appropriate tokenizer is used.

        For raw Greek text (no verse annotations), the entire text is
        split into sentences using Greek punctuation rules, and each
        sentence is stored as a pseudo-verse.
        """
        directory = Path(directory)
        epistles = []

        target_list = UNDISPUTED_EPISTLES if undisputed_only else ALL_EPISTLES

        for ep_name in target_list:
            # Try multiple filename patterns
            candidates = [
                directory / f"{ep_name}.txt",
                directory / f"{ep_name.replace(' ', '_')}.txt",
                directory / f"{PAULINE_EPISTLES[ep_name][0]}.txt",
            ]

            filepath = None
            for c in candidates:
                if c.exists():
                    filepath = c
                    break

            if filepath is None:
                logger.warning(f"No text file found for {ep_name}, skipping")
                continue

            text = filepath.read_text(encoding="utf-8")
            verses = _parse_text_to_verses(ep_name, text)
            if verses:
                epistles.append(Epistle(name=ep_name, verses=verses))
                logger.info(
                    f"  Loaded {ep_name}: {len(verses)} units, "
                    f"{sum(len(word_tokenize(v.text)) for v in verses)} words"
                )
            else:
                logger.warning(f"  No content parsed for {ep_name}")

        return cls(epistles=epistles, undisputed_only=undisputed_only)


def _parse_text_to_verses(book: str, text: str) -> list[Verse]:
    """
    Parse raw text into verse objects.

    Supports three formats:
    1. Verse-annotated: lines starting with 'chapter:verse' pattern
    2. Raw Greek text: split into sentences on Greek punctuation (. ;)
    3. Raw English text: split into sentences via NLTK
    """
    text = text.strip()
    if not text:
        return []

    verses: list[Verse] = []

    # Check for verse annotations first
    verse_pattern = re.compile(r"^(\d+):(\d+)\s+(.*)")
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = verse_pattern.match(line)
        if match:
            chapter, verse_num, verse_text = match.groups()
            verses.append(Verse(
                book=book,
                chapter=int(chapter),
                verse=int(verse_num),
                text=verse_text.strip(),
            ))

    if verses:
        return verses

    # No verse annotations — split into sentences as pseudo-verses
    sentences = sent_tokenize(text)
    for i, s in enumerate(sentences, 1):
        s = s.strip()
        if s:
            verses.append(Verse(book=book, chapter=1, verse=i, text=s))

    return verses
