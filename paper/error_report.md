# Error Report: Pauline Neural Study

Cross-referencing of `paper/study_design.md` against the full codebase in `src/pauline/`.

---

## 1. Paper Errors

### 1.1 Section 4.3 — Cross-Epistle Epistle Count (Internal Contradiction)

**Location in paper:** Section 4.3, paragraphs on "Leave-one-out" and "All-subsets"

**The text states:**
> Leave-one-out: For each epistle E_i, train embeddings on the remaining **6** epistles.

> All-subsets: Analyze all C(7,3) + C(7,4) + ... + C(7,7) = **99** subsets of 3 or more epistles.

**The problem:** This describes a 7-epistle (undisputed-only) design, but the rest of the paper consistently uses 14 epistles:

- Section 4.1 (Bootstrap): "Epistle-level (n = **14** units)"
- Section 5.2 (Corpus): "The corpus consists of **14** Pauline epistles in Koine Greek"
- Section 6 (Results): Reports per-epistle results for all 14 epistles including disputed letters and Hebrews
- Section 7.3 (Limitations): "With **14** epistles as the largest sampling unit" and "leave-one-out analysis (**15** subsets)"

**Correct values for 14 epistles:**
- Leave-one-out: remaining **13** epistles (not 6)
- All-subsets: C(14,3) + C(14,4) + ... + C(14,14) = **16,370** subsets (not 99)

**Classification:** Paper drafting error. Section 4.3 appears to describe an earlier version of the methodology that used only 7 undisputed epistles, while the actual analysis (and all results) used all 14.

---

## 2. Paper Omissions

### 2.1 Undocumented `.isalpha()` Vocabulary Filter

**Code locations:**
- `src/pauline/vae/model.py:419` — VAE vocabulary: `freq >= self.min_word_freq and w.isalpha()`
- `src/pauline/bayesian/model.py:208` — Bayesian vocabulary: `count >= 2 and w.isalpha()`
- `src/pauline/fractal/analyzer.py:573` — Multi-scale co-occurrence: `w.isalpha()`

**Paper states (Section 4.6):**
> vocabulary of 2,757 words (minimum frequency >= 2)

**The problem:** The paper describes only a frequency-based filter, but the code also applies `.isalpha()`, which excludes any token containing non-alphabetic characters (e.g., numeric strings, mixed alphanumeric tokens). This is an additional unstated vocabulary reduction step.

**Impact:** For Koine Greek, `.isalpha()` correctly accepts Greek letters (including polytonic diacriticals), so the practical impact is limited to excluding rare non-alphabetic tokens. However, the filter should be documented for reproducibility.

### 2.2 Ambiguous Final Sigma Claim

**Paper location:** Section 5.1
> Greek lowercasing correctly handles final sigma (ς vs. σ).

**Code location:** `src/pauline/corpus/loader.py:128-135`
```python
def normalize_greek(word: str) -> str:
    return unicodedata.normalize("NFC", word.lower())
```

**The nuance:** Python's `str.lower()`:
- **Preserves** existing ς (final sigma, U+03C2) and σ (medial sigma, U+03C3) as distinct characters — correct.
- **Converts** Σ (capital sigma, U+03A3) to σ (medial sigma) regardless of position — does NOT produce ς at word-final position.

The paper's claim is true in the sense that `lower()` does not collapse ς and σ into a single character. However, it could be misread as claiming that `lower()` correctly produces ς at word-final positions, which it does not. Since the Greek source texts already use proper final sigma in lowercase, this is not a practical issue, but the claim should be clarified.

---

## 3. Code Bugs

### 3.1 Default Pipeline Phases Missing 3 of 10 Phases

**File:** `src/pauline/config.py:165-173`

**Current default phases:**
```python
phases: list[str] = field(default_factory=lambda: [
    "corpus", "bootstrap", "embeddings", "cross_epistle",
    "combinatorial", "fractal", "analysis",
])
```

**Missing:** `"permutation"`, `"vae"`, `"bayesian"`

**Paper reference (Section 5.2):**
> The complete pipeline (10 phases, 1,000 bootstrap samples per level) ran on an e2-standard-4 VM

**Impact:** Running the pipeline with default configuration skips 3 phases. Users must explicitly add these phases via YAML config or --phases CLI flag to get the full 10-phase analysis described in the paper. The paper results cannot be reproduced with default settings.

**Fix:** Add the three missing phases to the default list.

### 3.2 `_is_greek` Operator Precedence Bug

**File:** `src/pauline/corpus/loader.py:83-84`

**Current code:**
```python
greek_chars = sum(1 for ch in text if unicodedata.category(ch).startswith("L")
                  and "\u0370" <= ch <= "\u03FF" or "\u1F00" <= ch <= "\u1FFF")
```

**Problem:** Due to Python operator precedence (`and` binds tighter than `or`), this is parsed as:
```python
(startswith("L") and "\u0370" <= ch <= "\u03FF") or ("\u1F00" <= ch <= "\u1FFF")
```

The `startswith("L")` category check is **bypassed** for characters in the Greek Extended range (U+1F00-U+1FFF). Any character in that range — including theoretically unassigned codepoints — would be counted as Greek without verifying it is actually a letter.

**Practical impact:** Low. All currently assigned codepoints in U+1F00-U+1FFF are Greek letters with diacriticals. However, the logic is incorrect and should use explicit parentheses.

**Fix:** Add parentheses: `startswith("L") and ("\u0370" <= ch <= "\u03FF" or "\u1F00" <= ch <= "\u1FFF")`

### 3.3 fetch.py Missing Hebrews

**File:** `src/pauline/corpus/fetch.py:46-77`

**Problem:** `BIBLE_API_BOOKS` (line 46) and `EPISTLE_CHAPTERS` (line 63) each list 13 epistles but do not include Hebrews. Meanwhile:
- `src/pauline/corpus/loader.py:70` includes `"Hebrews": ("Heb", False)` in `PAULINE_EPISTLES`
- `data/Hebrews.txt` exists in the data directory

**Impact:** The API fetcher (used for English/WEB text) cannot download Hebrews. This only affects the `source="api"` path; the default `source="text_files"` path loads Hebrews from the local file. Still, the inconsistency between fetch.py and loader.py could cause confusion.

**Fix:** Add Hebrews to both `BIBLE_API_BOOKS` and `EPISTLE_CHAPTERS`.

---

## Summary

| # | Type | Severity | Location |
|---|------|----------|----------|
| 1.1 | Paper error | High | Section 4.3 — says 7 epistles, should be 14 |
| 2.1 | Paper omission | Low | `.isalpha()` filter undocumented |
| 2.2 | Paper omission | Low | Final sigma claim ambiguous |
| 3.1 | Code bug | High | `config.py:165` — default phases missing 3/10 |
| 3.2 | Code bug | Low | `loader.py:83` — operator precedence |
| 3.3 | Code bug | Medium | `fetch.py:46,63` — Hebrews missing |
