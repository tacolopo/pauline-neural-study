"""
Corpus Fetcher
==============

Downloads Pauline epistle text from public domain sources.

Supported sources:
    - bible-api.com (free, no auth, World English Bible - public domain)
    - Local text files (user-provided, any translation/language)
    - Local JSON files (structured verse-level data)

The World English Bible (WEB) is used as the default because it is
a modern English translation in the public domain, making it suitable
for computational research without licensing concerns.

For Greek text (preferred for scholarly work), users should provide
their own Koine Greek text files from sources such as:
    - SBLGNT (Society of Biblical Literature Greek New Testament)
    - Nestle-Aland (NA28) - requires license
    - Westcott-Hort (1881) - public domain
    - Tischendorf (8th edition) - public domain
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from .loader import (
    PAULINE_EPISTLES,
    UNDISPUTED_EPISTLES,
    ALL_EPISTLES,
    Verse,
    Epistle,
    PaulineCorpus,
)

logger = logging.getLogger(__name__)

# bible-api.com book name mappings
BIBLE_API_BOOKS = {
    "Romans": "Romans",
    "1 Corinthians": "1 Corinthians",
    "2 Corinthians": "2 Corinthians",
    "Galatians": "Galatians",
    "Philippians": "Philippians",
    "1 Thessalonians": "1 Thessalonians",
    "Philemon": "Philemon",
    "Ephesians": "Ephesians",
    "Colossians": "Colossians",
    "2 Thessalonians": "2 Thessalonians",
    "1 Timothy": "1 Timothy",
    "2 Timothy": "2 Timothy",
    "Titus": "Titus",
}

# Chapter counts for each epistle
EPISTLE_CHAPTERS = {
    "Romans": 16,
    "1 Corinthians": 16,
    "2 Corinthians": 13,
    "Galatians": 6,
    "Philippians": 4,
    "1 Thessalonians": 5,
    "Philemon": 1,
    "Ephesians": 6,
    "Colossians": 4,
    "2 Thessalonians": 3,
    "1 Timothy": 6,
    "2 Timothy": 4,
    "Titus": 3,
}


class CorpusFetcher:
    """
    Fetches Pauline corpus text from public domain sources.

    Default source is bible-api.com which serves the World English Bible
    (public domain). Fetched data is cached locally as JSON for offline use.
    """

    BIBLE_API_BASE = "https://bible-api.com"

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.data_dir / "pauline_corpus.json"

    def fetch(
        self,
        undisputed_only: bool = True,
        force_refresh: bool = False,
        translation: str = "web",
    ) -> PaulineCorpus:
        """
        Fetch the Pauline corpus, using cache if available.

        Args:
            undisputed_only: Only include the 7 undisputed epistles.
            force_refresh: Re-download even if cache exists.
            translation: Bible translation to use (default: 'web' for
                        World English Bible).

        Returns:
            PaulineCorpus instance ready for analysis.
        """
        if self.cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached corpus from {self.cache_file}")
            return PaulineCorpus.from_json(self.cache_file, undisputed_only=undisputed_only)

        logger.info("Fetching Pauline corpus from bible-api.com...")
        epistles = self._fetch_from_api(
            undisputed_only=undisputed_only,
            translation=translation,
        )

        # Cache the fetched data
        self._save_cache(epistles)

        return PaulineCorpus(epistles=epistles, undisputed_only=undisputed_only)

    def _fetch_from_api(
        self,
        undisputed_only: bool = True,
        translation: str = "web",
    ) -> list[Epistle]:
        """Fetch epistles from bible-api.com chapter by chapter."""
        epistle_names = UNDISPUTED_EPISTLES if undisputed_only else ALL_EPISTLES
        epistles = []

        for ep_name in epistle_names:
            api_name = BIBLE_API_BOOKS[ep_name]
            num_chapters = EPISTLE_CHAPTERS[ep_name]
            all_verses: list[Verse] = []

            logger.info(f"  Fetching {ep_name} ({num_chapters} chapters)...")

            for chapter in range(1, num_chapters + 1):
                reference = f"{api_name} {chapter}"
                url = f"{self.BIBLE_API_BASE}/{reference}"
                params = {"translation": translation}

                try:
                    resp = requests.get(url, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()

                    for v in data.get("verses", []):
                        text = v.get("text", "").strip()
                        if text:
                            all_verses.append(Verse(
                                book=ep_name,
                                chapter=v.get("chapter", chapter),
                                verse=v.get("verse", 0),
                                text=text,
                            ))

                    # Rate limiting: be polite to the free API
                    time.sleep(0.5)

                except requests.RequestException as e:
                    logger.error(f"  Failed to fetch {reference}: {e}")
                    continue

            if all_verses:
                epistles.append(Epistle(name=ep_name, verses=all_verses))
                logger.info(
                    f"  {ep_name}: {len(all_verses)} verses, "
                    f"{sum(len(v.text.split()) for v in all_verses)} words"
                )
            else:
                logger.warning(f"  No verses fetched for {ep_name}")

        return epistles

    def _save_cache(self, epistles: list[Epistle]) -> None:
        """Save fetched corpus to JSON cache."""
        data = {
            "source": "bible-api.com",
            "translation": "World English Bible (WEB)",
            "license": "Public Domain",
            "epistles": [
                {
                    "name": ep.name,
                    "abbreviation": ep.abbreviation,
                    "is_undisputed": ep.is_undisputed,
                    "verses": [
                        {
                            "chapter": v.chapter,
                            "verse": v.verse,
                            "text": v.text,
                        }
                        for v in ep.verses
                    ],
                }
                for ep in epistles
            ],
        }

        with open(self.cache_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Corpus cached to {self.cache_file}")

    def load_local(
        self,
        directory: Optional[str | Path] = None,
        undisputed_only: bool = True,
    ) -> PaulineCorpus:
        """
        Load corpus from local text files.

        Place text files in the data directory named after each epistle:
            data/Romans.txt
            data/1_Corinthians.txt
            etc.

        Files should have verses in format:
            chapter:verse text content here
            1:1 Paul, a servant of Jesus Christ...
            1:2 To all who are in Rome...

        Or plain text (will be sentence-tokenized).
        """
        directory = Path(directory) if directory else self.data_dir
        return PaulineCorpus.from_text_files(directory, undisputed_only=undisputed_only)
