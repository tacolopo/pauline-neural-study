# Error Report: Pauline Neural Study

Cross-referencing of `paper/study_design.md` against the full codebase in `src/pauline/` and the GCP pipeline results from 2026-02-07.

---

## 1. Paper Errors

### 1.1 Section 4.3 — Cross-Epistle Epistle Count (Internal Contradiction)

**Location in paper:** Section 4.3, paragraphs on "Leave-one-out" and "All-subsets"

**The text stated:**
> Leave-one-out: For each epistle E_i, train embeddings on the remaining **6** epistles.

> All-subsets: Analyze all C(7,3) + C(7,4) + ... + C(7,7) = **99** subsets of 3 or more epistles.

**The problem:** This described a 7-epistle (undisputed-only) design, but the rest of the paper consistently uses 14 epistles:

- Section 4.1 (Bootstrap): "Epistle-level (n = **14** units)"
- Section 5.2 (Corpus): "The corpus consists of **14** Pauline epistles in Koine Greek"
- Section 6 (Results): Reports per-epistle results for all 14 epistles including disputed letters and Hebrews
- Section 7.3 (Limitations): "With **14** epistles as the largest sampling unit" and "leave-one-out analysis (**15** subsets)"

**Correct values for 14 epistles:**
- Leave-one-out: remaining **13** epistles (not 6)
- All-subsets: C(14,3) + C(14,4) + ... + C(14,14) = **16,370** subsets (not 99)

**Classification:** Paper drafting error. Section 4.3 described an earlier version of the methodology that used only 7 undisputed epistles.

**Status:** FIXED in paper.

### 1.2 Section 6.2 — Cross-Epistle Stability Values Were Stale

**Location in paper:** Section 6.2.1, stability table and prose

**The problem:** The paper reported specific stability values (e.g., ἀγάπη ↔ ἐλπίς = 0.982) from a previous run. The GCP run on 2026-02-07 produced slightly different values (e.g., 0.978) due to Word2Vec's stochastic training. The ranking of top pairs also shifted — ἁμαρτία ↔ ἐπαγγελία (sin ↔ promise) moved to #2, and δίκαιος ↔ δικαιοῦται (a morphological pair) appeared in the top 5.

**Key differences:**
| Pair | Old value | New value |
|------|-----------|-----------|
| ἀγάπη ↔ ἐλπίς | 0.982 | 0.978 |
| πίστις ↔ ἐπαγγελία | 0.976 | 0.973 |
| νόμου ↔ ἐπαγγελία | 0.973 | 0.964 |
| ἁμαρτία ↔ σάρξ | 0.971 | 0.963 |

**Impact:** Qualitative conclusions are unchanged — same hub word (ἐπαγγελία), same top pair (love-hope), same zero-stability δικαιοσύνη nominative pairs. Variation is ±0.005–0.010, consistent with Word2Vec stochasticity.

**Status:** FIXED — paper updated with new values and a note about stochastic variation.

### 1.3 Sections 5.2, 7.3, 8.2 — Incorrect Runtime Claims

**Location:** Section 5.2, Section 7.3 (contribution #7), Section 8.2

**The paper claimed:**
- Section 5.2: "approximately 5.5 hours"
- Section 7.3: "completing all 10 phases in **6.58 seconds**"
- Section 8.2: "The **6.58-second** runtime makes large-scale comparative studies practical"

**The problem:** The `pipeline_summary.json` from the GCP run shows `total_elapsed_seconds: 21654.98` (~6.02 hours) for only 7 phases (the 3 missing phases weren't run due to the config bug). The "6.58 seconds" claim appears to be from a `--quick` test run but was presented without that context, making it highly misleading.

**Status:** FIXED — paper updated to say "approximately 6 hours" for full run and notes that `--quick` mode enables rapid iteration.

---

## 2. Paper Omissions

### 2.1 Undocumented `.isalpha()` Vocabulary Filter

**Code locations:**
- `src/pauline/vae/model.py:419` — VAE vocabulary: `freq >= self.min_word_freq and w.isalpha()`
- `src/pauline/bayesian/model.py:208` — Bayesian vocabulary: `count >= 2 and w.isalpha()`
- `src/pauline/fractal/analyzer.py:573` — Multi-scale co-occurrence: `w.isalpha()`

**Paper states (Section 4.6):**
> vocabulary of 2,757 words (minimum frequency >= 2)

**The problem:** The paper describes only a frequency-based filter, but the code also applies `.isalpha()`, which excludes any token containing non-alphabetic characters. This is an additional unstated vocabulary reduction step.

**Impact:** Low. For Koine Greek, `.isalpha()` correctly accepts Greek letters (including polytonic diacriticals), so the practical impact is limited to excluding rare non-alphabetic tokens.

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

## 3. Code Bugs (Found and Fixed)

### 3.1 Default Pipeline Phases Missing 3 of 10 Phases

**File:** `src/pauline/config.py:165-173`

**Was:**
```python
phases: ["corpus", "bootstrap", "embeddings", "cross_epistle",
         "combinatorial", "fractal", "analysis"]
```

**Missing:** `"permutation"`, `"vae"`, `"bayesian"`

**Impact:** The GCP run on 2026-02-07 confirmed this — only 7 phases ran. The `bayesian_topics.json` in the results was from a prior run (file timestamp 16:13 predates the 16:24 pipeline start).

**Status:** FIXED — all three phases added to defaults.

### 3.2 `_is_greek` Operator Precedence Bug

**File:** `src/pauline/corpus/loader.py:83-84`

**Was:**
```python
greek_chars = sum(1 for ch in text if unicodedata.category(ch).startswith("L")
                  and "\u0370" <= ch <= "\u03FF" or "\u1F00" <= ch <= "\u1FFF")
```

Due to Python operator precedence (`and` binds tighter than `or`), the `startswith("L")` check was bypassed for the extended Greek range (U+1F00-U+1FFF).

**Status:** FIXED — added parentheses.

### 3.3 fetch.py Missing Hebrews

**File:** `src/pauline/corpus/fetch.py:46-77`

`BIBLE_API_BOOKS` and `EPISTLE_CHAPTERS` had 13 entries but not Hebrews, inconsistent with `loader.py` which includes Hebrews.

**Status:** FIXED — Hebrews added to both dicts.

### 3.4 TSNE `n_iter` Parameter Renamed in scikit-learn

**File:** `src/pauline/analysis/visualization.py:274, 831, 1473`

**The problem:** scikit-learn renamed `TSNE(n_iter=...)` to `TSNE(max_iter=...)`. The GCP run used scikit-learn 1.6+ which no longer accepts the old parameter name, causing the analysis phase to crash:
```
TypeError: TSNE.__init__() got an unexpected keyword argument 'n_iter'
```

**Impact:** High. The analysis phase (visualizations, t-SNE plots) failed entirely on the GCP run.

**Status:** FIXED — all 3 occurrences changed from `n_iter` to `max_iter`.

---

## 4. GCP Run Analysis (2026-02-07)

### 4.1 Run Summary

| Metric | Value |
|--------|-------|
| VM | e2-standard-4 (4 vCPUs, 16 GB RAM) |
| Start | 2026-02-07 16:24:55 UTC |
| End | 2026-02-07 22:25:51 UTC |
| Total | 21,655 seconds (~6.02 hours) |
| Phases run | 7 of 10 (missing permutation, vae, bayesian) |
| Python | 3.10.12 |
| scikit-learn | 1.6+ (TSNE API change) |
| gensim | 4.4.0 |

### 4.2 Phase Timings

| Phase | Duration | Notes |
|-------|----------|-------|
| corpus | 1.1s | 14 epistles, 37,235 words |
| bootstrap | 2,313.5s (~38.6 min) | 3 levels x 1,000 samples |
| embeddings | 19,257.5s (~5.35 hrs) | Dominates total runtime |
| cross_epistle | 61.3s (~1 min) | 15 subsets (leave-one-out) |
| combinatorial | 12.3s | 500 sentences generated |
| fractal | 6.1s | Hurst H=0.607 |
| analysis | FAILED | TSNE n_iter bug |

### 4.3 What Matched the Paper

- **Corpus stats**: 37,235 words, 7,815 vocab, 1,536 sentences, 14 epistles — exact match
- **Fractal Hurst exponent**: H = 0.607, R² = 0.999 — exact match (deterministic)
- **Per-epistle Hurst values**: All match (deterministic)
- **Bayesian topics**: Same 10-topic structure with matching per-epistle distributions (from prior run)
- **Cross-epistle qualitative patterns**: Same hub word (ἐπαγγελία), same top pair (love-hope), same zero-stability patterns for δικαιοσύνη nominative

### 4.4 What Differed

- **Cross-epistle specific values**: ±0.005–0.010 stochastic variation (expected, Word2Vec)
- **Ranking of top pairs**: Shifted slightly (ἁμαρτία ↔ ἐπαγγελία now #2, δίκαιος ↔ δικαιοῦται now #5)
- **Runtime**: ~6 hours for 7 phases (paper claimed 5.5 hours for 10 phases)
- **Analysis phase**: Failed due to TSNE API change

### 4.5 Recommendation

Re-run the pipeline after all code fixes (phases, TSNE) to produce a complete 10-phase result set. The current results are valid for the 7 phases that completed, but the permutation, VAE, and Bayesian phases were not re-run with the current codebase.

---

## Summary

| # | Type | Severity | Location | Status |
|---|------|----------|----------|--------|
| 1.1 | Paper error | High | Section 4.3 — said 7 epistles, should be 14 | FIXED |
| 1.2 | Paper error | Medium | Section 6.2 — stale stability values | FIXED |
| 1.3 | Paper error | High | Sections 5.2/7.3/8.2 — wrong runtime claims | FIXED |
| 2.1 | Paper omission | Low | `.isalpha()` filter undocumented | Documented |
| 2.2 | Paper omission | Low | Final sigma claim ambiguous | Documented |
| 3.1 | Code bug | High | `config.py:165` — default phases missing 3/10 | FIXED |
| 3.2 | Code bug | Low | `loader.py:83` — operator precedence | FIXED |
| 3.3 | Code bug | Medium | `fetch.py:46,63` — Hebrews missing | FIXED |
| 3.4 | Code bug | High | `visualization.py` — TSNE n_iter deprecated | FIXED |
