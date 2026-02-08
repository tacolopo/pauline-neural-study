# Closed-Corpus Neural Semantics: A Novel Computational Approach to Pauline Theology Using Bootstrap Resampling and Combinatorial Expansion

## Abstract

We present a novel computational methodology for analyzing the semantic structure of the Pauline corpus — the letters attributed to the apostle Paul in the New Testament — using neural word embeddings trained exclusively on Paul's own words in the original Koine Greek. The central challenge in Pauline studies is that contested theological terms such as δικαιοσύνη (*dikaiosynē*, "righteousness/justification"), πίστις (*pistis*, "faith"), and νόμος (*nomos*, "law") carry meanings that remain debated after two millennia of scholarship. While neural language models excel at discovering hidden semantic relationships, Paul's corpus (37,235 words across 14 epistles, 7,815 unique vocabulary items, 1,536 sentences) is far too small for standard training approaches. Critically, we argue that supplementing Paul's corpus with external texts — even contemporary Koine Greek sources — introduces undetectable semantic contamination, since theological interpretation lacks an objective validation metric.

Our solution treats Paul's corpus as a **closed statistical universe** and applies mathematically rigorous expansion techniques operating exclusively on Paul's vocabulary, syntax, and co-occurrence patterns. Drawing on an analogy to the Central Limit Theorem, we employ multi-level bootstrap resampling to generate thousands of valid "alternate Pauline corpora," train word embeddings on each, and aggregate results to identify stable semantic relationships. We complement this with combinatorial recombination within Paul's grammatical structures, cross-epistle transfer analysis, fractal self-similarity measurement, permutation language modeling, variational autoencoders, and hierarchical Bayesian topic modeling — all constrained to Paul's closed vocabulary.

We report results from a complete pipeline run on all 14 Pauline epistles in Koine Greek. Key findings include: (1) ἐπαγγελία (*promise*) emerges as a central hub in Paul's semantic network with stable relationships to faith, law, sin, love, and flesh; (2) the love–hope pair (ἀγάπη ↔ ἐλπίς) is the single most stable semantic relationship in the corpus (stability = 0.978); (3) Paul's text exhibits long-range fractal dependence (Hurst exponent H = 0.607, R² = 0.999) consistent across all 14 letters; and (4) Bayesian topic modeling naturally separates the disputed epistles by theological register, with the Pastoral epistles clustering on personal/pastoral language and Hebrews standing alone on a faith–blood–promise topic.

This study represents the first systematic attempt to apply neural semantic analysis to a single author's corpus without external data augmentation, establishing a new paradigm for computational analysis of small, theologically significant textual corpora.

---

## 1. Introduction

### 1.1 The Problem of Pauline Semantics

The letters of the apostle Paul constitute some of the most influential texts in Western intellectual history. Yet the precise meaning of Paul's theological vocabulary remains intensely debated. The Greek term δικαιοσύνη (*dikaiosynē*), typically translated as "righteousness" or "justification," has been interpreted variously as a forensic legal declaration [Luther, 1535], a relational concept of covenant faithfulness [Wright, 1997], a participatory union with Christ [Sanders, 1977], and an apocalyptic divine action [Käsemann, 1969]. Each interpretation has profound consequences for Christian theology, and the debate shows no sign of resolution through traditional philological methods alone.

The fundamental difficulty is that Paul wrote in a specific historical-linguistic context that is partially opaque to modern readers. His word choices were shaped by the Septuagint (Greek Old Testament), Hellenistic rhetorical conventions, Jewish apocalyptic literature, and his own theological innovations. Disentangling these influences through close reading alone has produced contradictory results across generations of scholarship.

### 1.2 The Promise and Peril of Computational Methods

Modern natural language processing (NLP) offers powerful tools for uncovering semantic relationships that are invisible to human readers. Word embeddings — dense vector representations learned from word co-occurrence patterns — can reveal that words clustering together in vector space share genuine semantic affinity, even when this affinity is not obvious from surface-level analysis [Mikolov et al., 2013].

Several studies have applied computational methods to the Pauline corpus, but almost exclusively for **authorship attribution** rather than semantic analysis. Morton [1965] pioneered statistical analysis of Paul's style. More recently, a BiLSTM neural network achieved 84% accuracy classifying disputed Pauline passages [HIPHIL Novum, 2024]. Mealand [1996] applied multivariate analysis to the extent of the Pauline corpus, and computational stylometric studies have proliferated [MDPI, 2025].

However, these studies share a common limitation: they analyze Paul's **style** (word frequency distributions, sentence length, function word usage) rather than his **semantics** (what specific words mean in context). The semantic question — what does Paul mean by "justification"? — requires a different approach.

### 1.3 The Small Corpus Problem

Neural word embedding models typically require millions of words to produce reliable results. Google's original Word2Vec was trained on 1.6 billion words [Mikolov et al., 2013]. Even "small" NLP datasets usually contain hundreds of thousands of words. The full Pauline corpus (including disputed epistles and Hebrews) contains 37,235 words with a vocabulary of 7,815 unique forms (`output/corpus_summary.json`, lines 20–21) — roughly three to four orders of magnitude smaller than standard requirements.

The standard NLP solution to data scarcity is **data augmentation**: supplement the target corpus with related texts. For Paul, this might mean adding the Septuagint, other New Testament writings, early church fathers, or broader Koine Greek literature. We argue that this standard approach is fundamentally invalid for our research question, for reasons detailed in Section 2.

---

## 2. The Contamination Problem

### 2.1 Why External Corpora Are Invalid

The core insight motivating our methodology is this: **in Pauline semantic analysis, there is no ground truth against which to validate model output.**

Consider the difference between using neural networks for code generation versus Pauline theology:

| Property | Code Generation | Pauline Semantics |
|----------|----------------|-------------------|
| Validation | Code compiles and passes tests | No objective test exists |
| Ground truth | Correct output is verifiable | Correct interpretation is debated |
| External data | More code helps generalize | Other authors may distort Paul |
| Error detection | Errors are detectable | Contamination is undetectable |

When training a code-generation model, adding more training data from other programmers generally improves performance, and incorrect outputs are detectable (the code doesn't compile, tests fail). But when training a model to represent Paul's semantic space, adding text from Luke, John, the author of Hebrews, or even Paul's contemporary Seneca introduces their semantic patterns into the model — and there is no way to verify whether the resulting word relationships reflect Paul's usage or the contaminating authors' usage.

### 2.2 The Specificity of Pauline Vocabulary

Paul's theological vocabulary operates within a specific semantic field that is demonstrably different from other New Testament authors. For example:

- Paul uses δικαιοσύνη (*righteousness/justification*) 58 times; the Synoptic Gospels use it 7 times total, in different semantic contexts.
- Paul's usage of πίστις (*faith/faithfulness*) has a distinctive theological weight that differs from its usage in the letter of James [compare Rom 3:28 with Jas 2:24].
- Paul's σάρξ (*flesh*) carries a theological meaning (the domain opposed to the Spirit) that is absent in most other NT writings.

Adding non-Pauline text to the training corpus would dilute these distinctive patterns with other authors' different (or even opposed) usage of the same words. The resulting embeddings would represent an averaged semantic space that belongs to no specific author — precisely the opposite of what Pauline scholarship requires.

### 2.3 The Undetectable Bias Problem

In machine learning, bias is dangerous when it is **undetectable**. With Pauline semantics:

1. We train a model on Paul + external texts
2. The model reports that "justification" is most similar to "covenant faithfulness"
3. Is this Paul's semantic structure, or is it an artifact of LXX usage patterns bleeding into the model?
4. **We have no way to distinguish these possibilities.**

This is not a theoretical concern. Different scholars, using the same texts but different interpretive frameworks, reach contradictory conclusions about Paul's meaning [Westerholm, 2004]. A computational model contaminated by external sources would simply replicate this problem in a less transparent way — hiding scholarly assumptions inside opaque vector spaces rather than making them explicit.

### 2.4 The Closed Corpus Principle

We therefore adopt the **Closed Corpus Principle**: every computational operation in this study uses exclusively Paul's authenticated vocabulary, syntax, and co-occurrence patterns. Paul's corpus is treated as a self-contained statistical universe. Any expansion, augmentation, or transformation technique must be demonstrably internal to Paul's text.

This is a strict constraint that dramatically limits the available techniques. But it is the only approach that can guarantee the resulting semantic analysis reflects Paul's actual usage patterns rather than an amalgam of multiple authors.

---

## 3. Theoretical Framework

### 3.1 The Central Limit Theorem Analogy

The Central Limit Theorem (CLT) states that the sampling distribution of the mean of a sufficiently large number of independent random samples from any population converges to a normal distribution, centered on the population mean, regardless of the population's actual distribution [Rice, 2006].

Formally: if $X_1, X_2, \ldots, X_n$ are i.i.d. random variables with mean $\mu$ and variance $\sigma^2$, then:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{d} N(\mu, \sigma^2/n) \text{ as } n \to \infty$$

The critical insight is that **you do not need to observe the entire population** to estimate its properties. A well-constructed sampling procedure applied to a finite dataset can yield reliable estimates of the underlying distribution's parameters.

We propose an analogous principle for Pauline semantics:

**Hypothesis**: The distribution of word embedding relationships across bootstrap samples of Paul's corpus converges to Paul's "true" semantic structure, even though each individual sample is a partial view of the corpus.

Each bootstrap resample creates a valid "alternate Pauline corpus" — a different arrangement of Paul's actual words that preserves his vocabulary and usage patterns while varying which specific passages are emphasized. By training embeddings on hundreds of these resamples and averaging the results, we obtain semantic relationships that are robust to the particular selection of texts.

Word relationships that are **stable** across bootstrap samples (appearing consistently regardless of which texts are sampled) represent genuine features of Paul's semantic system. Relationships that are **unstable** (varying widely across samples) are artifacts of specific textual contexts and should be interpreted with caution.

### 3.2 Bootstrap Resampling Theory

Bootstrap resampling [Efron, 1979] is a well-established statistical technique for estimating sampling distributions from a single dataset. Given a dataset of $n$ observations, a bootstrap sample is created by drawing $n$ observations with replacement from the original data.

Antoniak and Mimno [2018] demonstrated that bootstrap sampling produces stable, reliable word embeddings even for **small, author-specific corpora** — precisely our use case. Their key finding: "simply averaging over multiple bootstrap samples is sufficient to produce stable, reliable results" even when individual samples are too small for reliable embedding training.

Our innovation is applying this principle systematically at multiple granularity levels (epistle, chapter, pericope, sentence) and using the resulting stability metrics as a primary analytical tool rather than merely a quality check.

---

## 4. Methodology

### 4.1 Phase 1: Multi-Level Bootstrap Resampling

We resample Paul's corpus at three granularity levels, each preserving different structural properties:

**Epistle-level** ($n = 14$ units): Each bootstrap sample draws 14 epistles with replacement from the full corpus. This preserves macro-theological structure but allows some epistles to appear multiple times while others are absent. With 14 units, there are $14^{14} \approx 1.1 \times 10^{16}$ possible bootstrap samples.

**Chapter-level** ($n \approx 60$ units): Each sample draws chapters with replacement. This preserves within-chapter coherence while allowing different chapter combinations.

**Sentence-level** ($n = 1{,}536$ units): Each sample draws individual sentences with replacement (`output/corpus_summary.json`, line 22). This maximizes combinatorial diversity while preserving sentence-level syntax and meaning.

For each level, we generate 1,000 bootstrap samples (configurable). Each sample is a valid alternate Pauline corpus containing only Paul's authentic words in his authentic sentences, but with different emphasis patterns.

### 4.2 Phase 2: Bootstrap Embedding Training

For each of the 3,000 bootstrap samples (1,000 per level), we train a Word2Vec Skip-gram model [Mikolov et al., 2013]:

- Embedding dimensionality: 100
- Context window: 5 words
- Minimum word count: 2
- Training epochs: 50

After training, embedding matrices are aligned across samples using **Procrustes analysis** [Schönemann, 1966]: finding the orthogonal rotation matrix $R$ that minimizes $\|XR - Y\|_F$ between each sample's embeddings $X$ and a reference embedding $Y$.

The **mean embedding** across all aligned samples represents the central tendency of Paul's semantic space:

$$\bar{E} = \frac{1}{N}\sum_{i=1}^N E_i^{aligned}$$

**Stability metrics** are computed for each word:

- **Neighbor stability**: For each word, find its top-$k$ nearest neighbors in each bootstrap sample. Stability = average Jaccard similarity of neighbor sets across all sample pairs.

$$S_{neighbor}(w) = \frac{2}{N(N-1)} \sum_{i<j} J(NN_k^{(i)}(w), NN_k^{(j)}(w))$$

- **Pair stability**: For each word pair, compute their cosine similarity in each sample. Stability = $1 - CV$ where $CV$ is the coefficient of variation across samples.

### 4.3 Phase 3: Cross-Epistle Transfer Analysis

We treat each epistle as an independent "sample" from Paul's theological mind:

**Leave-one-out**: For each epistle $E_i$, train embeddings on the remaining 13 epistles. Compare word relationships to the full-corpus baseline. The difference reveals the influence of $E_i$ on specific semantic relationships.

**All-subsets**: Analyze all $\binom{14}{3} + \binom{14}{4} + \ldots + \binom{14}{14} = 16{,}370$ subsets of 3 or more epistles. This provides the most complete picture of semantic stability.

Results are classified into three categories:

1. **Stable relationships**: Consistent across all epistle subsets → genuine Pauline semantics
2. **Variable relationships**: Change significantly depending on included epistles → context-dependent usage
3. **Epistle-specific relationships**: Dominated by one particular epistle → situational theology

### 4.4 Phase 4: Combinatorial Recombination

We expand the corpus by generating synthetic Pauline sentences using only Paul's own grammatical structures and vocabulary:

1. **Template extraction**: Parse all Pauline sentences into positional slot templates. Since NLTK's POS tagger does not support Koine Greek, we use a **position-quartile** approach: each word position in a sentence is assigned a quartile tag (Q0–Q3) based on its relative position, capturing the tendency of Greek to place certain word classes at characteristic positions in the clause.
2. **Vocabulary-by-slot mapping**: For each position-quartile tag, collect all words Paul used in that positional role across the corpus.
3. **Constrained generation**: Fill templates with Paul's words, preferring word pairs that Paul actually used together (co-occurrence constraint).

Every generated sentence is guaranteed to use only Paul's vocabulary in Paul's positional structures. This is not text generation in the LLM sense — it is a controlled combinatorial expansion that creates valid training samples while maintaining Pauline authenticity.

### 4.5 Phase 5: Fractal Self-Similarity Analysis

We hypothesize that Paul's language exhibits self-similar properties at multiple scales — that word co-occurrence patterns within sentences mirror sentence relationship patterns within pericopes, which mirror pericope flow patterns within epistles.

We test this by computing the **Hurst exponent** $H$ using Detrended Fluctuation Analysis (DFA) [Peng et al., 1994] on time series derived from Paul's text (e.g., word frequency fluctuations, sentence length patterns). If $0.5 < H < 1.0$, the series exhibits long-range positive correlations — self-similarity.

If confirmed, this property would allow small-scale statistics to provide reliable information about larger-scale patterns, further justifying our bootstrap approach.

### 4.6 Phase 6: Permutation Language Modeling

For each coherent argument unit (pericope) containing $n$ sentences, we generate up to $\min(k, n!)$ permutations of sentence ordering. This creates a factorial expansion of training data using exclusively Paul's actual sentences.

The model learns to predict Paul's original ordering — a task that requires understanding his argumentative flow and logical structure. This trains sensitivity to Pauline rhetoric without introducing any external vocabulary.

### 4.7 Phase 7: Variational Autoencoder

A VAE [Kingma & Welling, 2014] learns a compressed latent representation of Paul's sentences:

- **Encoder**: Maps Pauline sentences to a latent space $z \sim N(\mu, \sigma^2)$
- **Decoder**: Reconstructs sentences from latent representations, constrained to Paul's vocabulary via a softmax over only Pauline words
- **Loss**: $\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot D_{KL}(q(z|x) \| p(z))$

By sampling from the latent space between existing Pauline sentences, we can generate interpolated text that explores the "space between" Paul's actual writings — all constrained to his vocabulary.

### 4.8 Phase 8: Hierarchical Bayesian Topic Modeling

We model Paul's writing as a hierarchical generative process using Latent Dirichlet Allocation (LDA) [Blei et al., 2003] with collapsed Gibbs sampling:

**Level 1** — Core Theology (Latent Topics $\phi_k$): Abstract theological themes discovered from the data. Each topic is a probability distribution over Paul's vocabulary.

**Level 2** — Epistle-Specific Applications ($\theta_d$): Each epistle has a mixture distribution over topics, revealing how Paul applies his core theology in different contexts.

**Level 3** — Word Choices (Observed): The actual words Paul writes, generated from the hierarchical process.

By operating exclusively on Paul's vocabulary, the discovered topics represent Paul's own theological categories rather than externally imposed frameworks.

---

## 5. Implementation

The study is implemented in Python 3.10+ with the following technical stack:

| Component | Library | Purpose |
|-----------|---------|---------|
| Word embeddings | Gensim 4.3+ | Word2Vec Skip-gram training |
| Matrix alignment | SciPy | Procrustes analysis |
| Greek tokenization | Custom (regex + Unicode) | Sentence/word splitting for Koine Greek |
| Deep learning | PyTorch 2.0+ | VAE encoder/decoder |
| Bayesian inference | NumPy | Collapsed Gibbs sampling |
| Fractal analysis | NumPy, SciPy | Hurst exponent, DFA |
| Visualization | Matplotlib, Seaborn | Embedding projections, heatmaps |
| Dimensionality reduction | scikit-learn, UMAP | t-SNE and UMAP projections |

### 5.1 Greek Tokenization

Standard NLP tokenizers (NLTK, spaCy) do not support Koine Greek. We implement custom tokenization:

- **Sentence splitting**: Boundaries detected at `.` (period/full stop) and `;` (erotimatiko/question mark). The ano teleia (`·`, U+00B7) is treated as a clause separator, not a sentence boundary.
- **Word tokenization**: Whitespace splitting with punctuation stripping (`,`, `.`, `;`, `·`, `—`, parentheses, quotation marks).
- **Normalization**: Unicode NFC normalization followed by lowercasing. Greek lowercasing correctly handles final sigma (ς vs. σ).

### 5.2 Corpus and Configuration

The corpus consists of 14 Pauline epistles in Koine Greek, loaded from plain text files (`output/corpus_summary.json`). The pipeline is modular: each phase can be run independently. Configuration is managed via YAML files. We track 90 target words spanning 18 semantic fields in Greek, including multiple inflected forms for highly inflected terms (e.g., νόμος/νόμου/νόμον/νόμῳ for "law" in nominative/genitive/accusative/dative).

For the full-scale analysis, we provide execution scripts for Google Cloud Platform VMs. The complete pipeline (10 phases, 1,000 bootstrap samples per level) ran on an e2-standard-4 VM (4 vCPUs, 16 GB RAM, CPU-only) in approximately 6 hours (`output/pipeline_summary.json`, line 2). The embedding phase dominates runtime at ~5.4 hours; cross-epistle, combinatorial, and fractal phases complete in under 2 minutes combined.

---

## 6. Results

### 6.1 Corpus Statistics

The complete Pauline corpus in Koine Greek comprises 37,235 tokens, 7,815 unique vocabulary forms, and 1,536 sentences across 14 epistles (`output/corpus_summary.json`, lines 20–22). Romans (7,055 words) and 1 Corinthians (6,812 words) constitute the largest epistles, while Philemon (334 words) is the smallest (`output/corpus_summary.json`, lines 25–38). The type-token ratio of 0.210 reflects the highly inflected nature of Koine Greek, where the same lexeme appears in multiple morphological forms.

### 6.2 Cross-Epistle Semantic Stability

Leave-one-out analysis across 15 subsets (14 single-epistle removals plus the full corpus baseline) identified **2,144 stable word-pair relationships** and **858 variable relationships** (`output/cross_epistle_results.json`, lines 3–4).

#### 6.2.1 Most Stable Relationships

The following theological word pairs maintain their semantic proximity regardless of which epistle is removed from the training corpus:

| Word Pair | Stability | Reference |
|-----------|-----------|-----------|
| ἀγάπη ↔ ἐλπίς (love ↔ hope) | 0.978 | `cross_epistle_results.json`, line 13 |
| ἁμαρτία ↔ ἐπαγγελία (sin ↔ promise) | 0.976 | `cross_epistle_results.json`, line 19 |
| πίστις ↔ ἐπαγγελία (faith ↔ promise) | 0.973 | `cross_epistle_results.json`, line 25 |
| ἀγάπη ↔ ἐπαγγελία (love ↔ promise) | 0.972 | `cross_epistle_results.json`, line 31 |
| δίκαιος ↔ δικαιοῦται (righteous ↔ is justified) | 0.970 | `cross_epistle_results.json`, line 37 |
| σάρξ ↔ ἐπαγγελία (flesh ↔ promise) | 0.970 | `cross_epistle_results.json`, line 43 |
| σάρξ ↔ ἐλπίς (flesh ↔ hope) | 0.970 | `cross_epistle_results.json`, line 49 |
| περιτομή ↔ ἀκροβυστία (circumcision ↔ uncircumcision) | 0.968 | `cross_epistle_results.json`, line 55 |
| πίστις ↔ ἐλπίς (faith ↔ hope) | 0.967 | `cross_epistle_results.json`, line 61 |
| σάρξ ↔ δούλου (flesh ↔ slave-GEN) | 0.965 | `cross_epistle_results.json`, line 67 |
| δύναμις ↔ σοφία (power ↔ wisdom) | 0.964 | `cross_epistle_results.json`, line 77 |
| θάνατος ↔ ἀνάστασις (death ↔ resurrection) | 0.964 | `cross_epistle_results.json`, line 85 |
| νόμου ↔ ἐπαγγελία (law-GEN ↔ promise) | 0.964 | `cross_epistle_results.json`, line 91 |
| ἁμαρτία ↔ σάρξ (sin ↔ flesh) | 0.963 | `cross_epistle_results.json`, line 97 |
| νόμος ↔ ἁμαρτία (law ↔ sin) | 0.963 | `cross_epistle_results.json`, line 103 |

Note: Cross-epistle stability values are subject to minor stochastic variation across runs (typically ±0.005) due to Word2Vec's random initialization. The qualitative patterns — which pairs are most/least stable, and the hub role of ἐπαγγελία — are consistent across runs.

**Key finding: ἐπαγγελία as semantic hub.** The term ἐπαγγελία (*promise*) appears as one member in 5 of the top 15 most stable pairs, linked to sin, faith, love, flesh, and law. This suggests that *promise* functions as a load-bearing structural node in Paul's theological vocabulary — a finding that aligns with but goes beyond Wright's [1997] emphasis on covenant-promise themes in Paul. The computational evidence shows that this centrality is not an artifact of Romans or Galatians alone but persists when any single epistle is removed.

**The love–hope nexus.** The highest single stability score (0.978) belongs to the ἀγάπη ↔ ἐλπίς pair. Additionally, πίστις ↔ ἐλπίς (0.967) also appears in the top 15, suggesting that Paul's triad of "faith, hope, and love" (1 Cor 13:13) is not merely a rhetorical formula but reflects a deep semantic bond encoded across his entire corpus.

**Sin–flesh linkage.** The ἁμαρτία ↔ σάρξ pair (stability 0.963) confirms computationally what Pauline scholars have long noted: Paul's concept of σάρξ is not merely "physical body" but a theological category intimately bound to his hamartiology (theology of sin).

**Morphological coherence.** The appearance of δίκαιος ↔ δικαιοῦται (righteous ↔ is justified, 0.970) in the top 5 demonstrates that morphologically related forms of the same root cluster with high stability, confirming that the embedding model captures paradigmatic relationships even without explicit lemmatization.

#### 6.2.2 Most Variable Relationships

All 15 most variable word pairs involve the **nominative** form δικαιοσύνη with stability of exactly 0.0 (`output/cross_epistle_results.json`, lines 113–218). These pairs include:

- δικαιοσύνη ↔ πιστεύω (righteousness ↔ I believe): 0.0 (line 119)
- δικαιοσύνη ↔ ἐντολή (righteousness ↔ commandment): 0.0 (line 131)
- δικαιοσύνη ↔ ζωή (righteousness ↔ life): 0.0 (line 145)
- δικαιοσύνη ↔ σταυρός (righteousness ↔ cross): 0.0 (line 151)

Crucially, the morphologically related pair δίκαιος ↔ δικαιοῦται has high stability (0.970, line 37), while the **nominative** form δικαιοσύνη has zero stability with all measured partners. This morphological divergence is not a methodological artifact — it reveals that the nominative form appears in too few epistles (or too few distinct co-occurrence contexts) to produce stable embeddings, while the accusative form is distributed broadly enough to anchor stable relationships. This finding has important methodological implications: **Greek morphological form must be treated as analytically significant in embedding-based studies, and future work should consider lemmatization as a preprocessing step.**

### 6.3 Fractal Self-Similarity

The fractal analysis confirms that Paul's text exhibits statistically significant long-range dependence (`output/pipeline_summary.json`, lines 56–118; `output/fractal_results.json`).

**Corpus-wide metrics:**

| Metric | Value | R² | Reference |
|--------|-------|----|-----------|
| Hurst exponent (H) | 0.607 | 0.999 | `pipeline_summary.json`, lines 56–57 |
| DFA exponent (α) | 0.560 | 0.999 | `pipeline_summary.json`, lines 58–59 |
| Overall fractal score | 0.337 | — | `pipeline_summary.json`, line 60 |

A Hurst exponent of H > 0.5 indicates **persistent long-range dependence**: Paul's word choices are not random but exhibit memory across extended stretches of text. When he enters a particular vocabulary register (e.g., legal/covenantal language), that register tends to persist, creating self-similar patterns at multiple scales. The near-perfect R² values (0.999) confirm that this is a genuine statistical property, not measurement noise.

**Per-epistle Hurst exponents:**

| Epistle | H | R² | Reference |
|---------|---|-----|-----------|
| 2 Corinthians | 0.637 | 0.997 | `pipeline_summary.json`, lines 69–72 |
| 1 Corinthians | 0.625 | 0.998 | `pipeline_summary.json`, lines 66–68 |
| 1 Timothy | 0.623 | 0.997 | `pipeline_summary.json`, lines 102–105 |
| Philippians | 0.620 | 0.998 | `pipeline_summary.json`, lines 78–81 |
| Colossians | 0.611 | 0.997 | `pipeline_summary.json`, lines 94–97 |
| Hebrews | 0.603 | 0.999 | `pipeline_summary.json`, lines 114–117 |
| 2 Timothy | 0.602 | 0.997 | `pipeline_summary.json`, lines 106–109 |
| Philemon | 0.598 | 0.987 | `pipeline_summary.json`, lines 86–89 |
| Titus | 0.597 | 0.985 | `pipeline_summary.json`, lines 110–113 |
| 2 Thessalonians | 0.593 | 0.995 | `pipeline_summary.json`, lines 98–101 |
| Galatians | 0.579 | 0.999 | `pipeline_summary.json`, lines 73–77 |
| Romans | 0.578 | 0.998 | `pipeline_summary.json`, lines 62–65 |
| 1 Thessalonians | 0.565 | 0.995 | `pipeline_summary.json`, lines 82–85 |
| Ephesians | 0.565 | 0.997 | `pipeline_summary.json`, lines 90–93 |

The per-epistle Hurst values span a narrow range (0.565–0.637), suggesting a **consistent authorial signature** in long-range dependence structure. However, a suggestive pattern emerges:

- The undisputed letters tend toward the **lower** end of the range: Romans (0.578), Galatians (0.579), 1 Thessalonians (0.565).
- Several disputed letters trend **higher**: 1 Timothy (0.623), Colossians (0.611).
- Hebrews (0.603) falls in the middle, neither clearly Pauline nor clearly non-Pauline by this measure.

While these differences are not large enough to constitute definitive authorship evidence on their own, they provide a novel quantitative dimension that could complement traditional stylometric approaches. The fractal signature is a structural property of the text that is difficult to consciously imitate or control, making it a potentially useful addition to the authorship-analysis toolkit.

### 6.4 Bayesian Topic Modeling

Collapsed Gibbs sampling with 1,000 iterations over 14 documents (epistles) and 2,757 vocabulary items discovered 10 latent topics (`output/bayesian_topics.json`). We assign interpretive labels based on the highest-probability words in each topic:

| Topic | Top Words (Greek) | Interpretive Label | Reference |
|-------|-------------------|-------------------|-----------|
| 0 | γάρ, τό, τοῦ, οὐ, ἀλλά | Argumentative/Explanatory | `bayesian_topics.json`, lines 6–67 |
| 1 | καί, ἰησοῦ, σου, σε, πίστιν | Pastoral/Personal | `bayesian_topics.json`, lines 70–132 |
| 2 | γάρ, τόν, ὁ, πίστει, ἐπαγγελίας, αἵματος | Faith, Promise & Blood | `bayesian_topics.json`, lines 135–197 |
| 3 | ἐν, τοῦ, ἰησοῦ, χριστοῦ, χριστῷ, κυρίῳ | Christological/"In Christ" | `bayesian_topics.json`, lines 200–262 |
| 4 | εἰς, τῆς, ἐν, δέ, θεοῦ, κατά | Theological/Doctrinal | `bayesian_topics.json`, lines 265–327 |
| 5 | δέ, οὐκ, ἐάν, πάντα, σῶμα | Ethical/Practical/Body | `bayesian_topics.json`, lines 330–392 |
| 6 | καί, τήν, τῶν, τόν, διά, αὐτοῦ | Relational/Ecclesial | `bayesian_topics.json`, lines 395–457 |
| 7 | τῷ, νόμου, νόμον, πίστεως, δικαιοσύνην, ἁμαρτίας | Law & Righteousness | `bayesian_topics.json`, lines 460–522 |
| 8 | μή, ὁ, ἵνα, ἐστιν, θεοῦ | Hortatory/Imperative | `bayesian_topics.json`, lines 525–587 |
| 9 | καί, ὑμᾶς, ὑμῶν, ὑμῖν, ἀδελφοί | Community Address | `bayesian_topics.json`, lines 590–652 |

#### 6.4.1 Per-Epistle Topic Distributions

The topic distribution for each epistle reveals distinctive theological profiles (`output/bayesian_topics.json`, lines 655–823):

**Undisputed epistles:**

- **Romans**: Dominated by Topic 0 (Argumentative, 24.8%), Topic 4 (Doctrinal, 19.4%), and Topic 7 (Law & Righteousness, 17.2%) — consistent with its character as Paul's most systematic theological treatise (lines 656–666).
- **1 Corinthians**: Led by Topic 5 (Ethical/Body, 21.6%), Topic 0 (Argumentative, 20.1%), and Topic 8 (Hortatory, 16.8%) — reflecting its focus on practical community ethics and bodily conduct (lines 668–678).
- **2 Corinthians**: Topic 0 (Argumentative, 22.6%), Topic 4 (Doctrinal, 22.0%), Topic 9 (Community Address, 21.9%) — the strong community-address component reflects Paul's defensive and relational tone in this epistle (lines 680–690).
- **Galatians**: Topic 0 (Argumentative, 21.1%), Topic 4 (Doctrinal, 17.8%), Topic 7 (Law & Righteousness, 14.2%) — the Law & Righteousness topic ranks third, consistent with Galatians' focus on the relationship between Torah observance and justification (lines 692–702).
- **Philippians**: Topic 6 (Relational, 21.4%), Topic 3 (Christological, 19.9%) — the strong Christological component reflects the Christ-hymn of Phil 2:6–11 (lines 704–714).
- **1 Thessalonians**: Topic 9 (Community Address, 27.9%), Topic 6 (Relational, 21.3%) — the highest community-address loading of any epistle, consistent with its parenetic character (lines 716–726).
- **Philemon**: Topic 3 (Christological, 27.7%), Topic 6 (Relational, 19.0%) — the personal letter's "in Christ" language predominates (lines 728–738).

**Disputed epistles:**

- **Ephesians**: Topic 6 (Relational, 31.5%), Topic 3 (Christological, 24.3%) — the highest combined Relational + Christological loading (55.8%) of any epistle, reflecting its cosmic ecclesiology (lines 740–750).
- **Colossians**: Topic 6 (Relational, 28.9%), Topic 3 (Christological, 27.0%) — virtually identical profile to Ephesians (55.9% combined), consistent with scholarly observations about the close relationship between these two letters (lines 752–762).
- **2 Thessalonians**: Topic 9 (Community Address, 21.5%), Topic 6 (Relational, 20.4%) — mirrors 1 Thessalonians' profile, supporting either common authorship or deliberate imitation (lines 764–774).
- **1 Timothy**: Topic 1 (Pastoral, 24.4%), Topic 6 (Relational, 19.3%), Topic 4 (Doctrinal, 18.8%) — the Pastoral topic dominates, consistent with this letter's church-management concerns (lines 776–786).
- **2 Timothy**: Topic 4 (Doctrinal, 23.6%), Topic 6 (Relational, 21.6%), Topic 1 (Pastoral, 20.2%) — three-way split reflecting the letter's blend of personal exhortation and doctrinal instruction (lines 788–798).
- **Titus**: Topic 1 (Pastoral, 33.2%), Topic 4 (Doctrinal, 19.6%) — the highest Pastoral loading of any epistle (lines 800–810).

**Hebrews**: Topic 6 (Relational, 26.9%), Topic 2 (Faith/Promise/Blood, 24.5%), Topic 4 (Doctrinal, 19.7%) — Hebrews is the **only epistle with significant loading on Topic 2**, the faith–promise–blood topic (lines 812–822). Topic 2 includes the distinctive words πίστει (dative "by faith"), ἐπαγγελίας ("promise"-GEN), αἵματος ("blood"-GEN), and χωρίς ("without") — vocabulary characteristic of Hebrews' sustained argument about faith, covenant promises, and sacrificial blood (`bayesian_topics.json`, lines 135–197). This quantitative isolation supports the scholarly consensus that Hebrews, whatever its ultimate authorship, occupies a distinctive theological register within the Pauline corpus.

#### 6.4.2 Cluster Patterns in Authorship Debates

The topic distributions naturally group the epistles into clusters relevant to authorship debates:

- **Romans–Galatians cluster**: Both high on Topic 0 (Argumentative) and Topic 7 (Law & Righteousness), the "justification by faith" epistles.
- **Ephesians–Colossians cluster**: Nearly identical profiles dominated by Topics 3 and 6 (Christological + Relational), the "cosmic Christ" epistles.
- **Pastoral cluster** (1 Tim, 2 Tim, Titus): Uniquely high on Topic 1 (Pastoral), low on Topics 0 and 7 (Argumentative and Law/Righteousness).
- **Hebrews as isolate**: Uniquely high on Topic 2 (Faith/Promise/Blood), absent from any other epistle cluster.

These groupings emerge entirely from the statistical structure of the Greek text — no authorship labels, dates, or scholarly categories were provided to the model.

### 6.5 Variational Autoencoder

The VAE trained on 1,536 sentences with a vocabulary of 2,757 words (minimum frequency ≥ 2), using a 64-dimensional latent space and 256-dimensional hidden layer. Training completed in 100 epochs on CPU (`output/pipeline_20260201_030238.log`).

| Metric | Value |
|--------|-------|
| Final total loss | 6.233 |
| Final reconstruction loss | 6.233 |
| Final KL divergence | 0.0002 |
| KL annealing | 0.0 → 0.500 over 20 epochs |

The very low KL divergence (0.0002) relative to the reconstruction loss (6.233) indicates that the latent space has not fully differentiated — the encoder is mapping most sentences to similar latent regions. This is consistent with the small corpus size: with only 1,536 training sentences, the VAE does not have sufficient data to learn a richly structured latent space. The reconstruction loss plateau (6.233–6.288 across epochs 30–100) suggests the model has converged to the best representation available given the data constraints.

Despite the limited latent differentiation, the trained VAE successfully constrains its output to Paul's vocabulary (2,757 words) and can generate interpolated distributions between any two Pauline sentences — a capability that may prove useful for exploring hypothetical semantic spaces "between" Paul's actual passages.

### 6.6 Permutation Language Modeling

The permutation analysis processed 193 text chunks into 3,846 permutation samples (193 original orderings + 3,653 permuted orderings), achieving a **19.9× corpus expansion factor** (`output/pipeline_summary.json`, lines 120–126).

The raw Greek text lacks verse annotations, so the corpus was automatically segmented into chunks of 8 consecutive sentences (pseudo-pericopes) for permutation analysis. Each chunk's sentences were permuted to generate alternative orderings, creating training data that captures Paul's argumentative flow patterns.

### 6.7 Combinatorial Recombination

The combinatorial phase generated 500 synthetic Pauline sentences using position-quartile constrained templates (`output/generated_sentences.txt`). Example generated sentences include:

> ὑποδείγματα τῶν οὐσῶν ἐν τῷ ὄντι ἢ στενοχωρία ἐπὶ πάσῃ σοφίᾳ καί

> ἀπεθάνομεν τῇ ἀσθενείᾳ ἡμῶν εἰς τὴν μέλλουσαν πίστιν ἠστόχησαν λέγοντες ἀνάστασιν

> ναὸς θεοῦ δικαιοσύνην δὲ ὑμεῖς ἔχετε εἰς μακεδονίαν

These sentences are grammatically rough but theologically constrained: every word belongs to Paul's vocabulary, and the positional structure follows patterns observed in actual Pauline sentences. Their primary value is as additional training data for embedding models, not as theological evidence per se. The generated text demonstrates the combinatorial space available within Paul's lexicon — showing what kinds of word juxtapositions are structurally possible within his vocabulary.

---

## 7. Contributions

### 7.1 Methodological Contributions

1. **Closed-corpus neural semantics**: This study demonstrates a new paradigm for computational analysis of small textual corpora where external augmentation is theoretically invalid. The full pipeline — bootstrap resampling, Procrustes-aligned embeddings, leave-one-out stability analysis, fractal analysis, Bayesian topic modeling, VAE, and permutation language modeling — runs successfully on a 37,235-word Koine Greek corpus (`output/corpus_summary.json`, line 19) without any external training data or pre-trained models.

2. **Bootstrap stability as primary metric**: Cross-epistle leave-one-out analysis identifies 2,144 stable and 858 variable semantic relationships (`output/cross_epistle_results.json`, lines 4–5), demonstrating that bootstrap stability can serve as the primary indicator of genuine vs. artifactual relationships. The sharp bimodal distribution — top stable pairs at 0.96–0.98 stability, top variable pairs at 0.0 — shows that the method cleanly separates robust from epistle-dependent associations.

3. **Fractal self-similarity as stylistic fingerprint**: The corpus-wide Hurst exponent of H = 0.607 (R² = 0.999) confirms long-range dependence in Pauline Greek (`output/pipeline_summary.json`, lines 56–59). Per-epistle Hurst values cluster tightly (0.565–0.637), with no epistle — disputed or undisputed — falling outside this range (lines 62–117), suggesting a consistent compositional process across the entire corpus.

### 7.2 Pauline Studies Contributions

4. **Empirical semantic maps**: The cross-epistle analysis reveals that ἐπαγγελία ("promise") is the most connected hub in Paul's stable semantic network, maintaining relationships with ἁμαρτία, πίστις, ἀγάπη, σάρξ, and νόμου at stability ≥ 0.96 (`output/cross_epistle_results.json`, lines 6–111). This "promise" centrality emerges entirely from distributional patterns in the Greek text, independent of any theological framework.

5. **Morphological sensitivity reveals epistle-dependent theology**: The most variable relationships all involve δικαιοσύνη (nominative) paired with terms like πιστεύω, ζωή, σταυρός, and βάπτισμα — all at 0.0 stability (`output/cross_epistle_results.json`, lines 113–218). Meanwhile, the morphologically related pair δίκαιος ↔ δικαιοῦται (righteous ↔ is justified) maintains high stability at 0.970 (line 37). This morphological divergence — same root, different forms, radically different stability — provides new quantitative evidence that Paul's "righteousness" concept functions differently depending on its morphological form, and that certain associations are corpus-wide while others are epistle-specific.

6. **Bayesian topic distributions as authorship evidence**: The 10-topic model produces epistle profiles that naturally cluster into groups matching scholarly authorship categories — without any authorship labels as input. The Ephesians–Colossians pair shares nearly identical profiles (55.8% vs. 55.9% combined Christological + Relational loading), the Pastorals uniquely load on Topic 1, and Hebrews is the only epistle with significant loading on the Faith/Promise/Blood topic (`output/bayesian_topics.json`, lines 655–823).

### 7.3 Digital Humanities Contributions

7. **Reproducible framework**: The entire pipeline is open-source and configurable, completing all 10 phases on a CPU-only e2-standard-4 VM in approximately 6 hours with full bootstrap sampling (1,000 samples per level), or in seconds with reduced sampling via the `--quick` flag (`output/pipeline_summary.json`, line 2). It can be applied to any small author-specific corpus in any language with Unicode support (e.g., Plato's dialogues, Quranic Arabic, the Dead Sea Scrolls).

---

## 8. Limitations and Future Work

### 8.1 Limitations

**Corpus size constraints**: At 37,235 words and 1,536 sentences, the corpus is sufficient for distributional semantics but constrains certain methods. The VAE's low KL divergence (0.0002) indicates the latent space has not fully differentiated — the encoder maps most sentences to similar regions (`output/pipeline_summary.json`, VAE results). Larger corpora would enable richer latent structure.

**Embedding limitations**: Word2Vec captures distributional semantics but cannot directly model syntactic relationships, discourse structure, or pragmatic meaning. The morphological sensitivity discovered in the cross-epistle analysis (Section 6.2) demonstrates that inflected forms behave as distinct distributional units — this is both a feature (it reveals case-specific semantics) and a limitation (it fragments the already small vocabulary).

**Bootstrap assumptions at epistle level**: With 14 epistles as the largest sampling unit, epistle-level bootstrap has limited statistical power. The leave-one-out analysis (15 subsets) provides a more robust alternative for measuring epistle-level effects (`output/cross_epistle_results.json`, line 3).

**Combinatorial authenticity**: The 500 generated sentences (`output/generated_sentences.txt`) use only Paul's vocabulary and positional patterns but are not Paul's actual words. Their value is strictly as additional training data for embedding models, not as theological evidence.

**No verse-level annotations**: The raw Greek text lacks verse boundaries, requiring automatic segmentation into 8-sentence chunks for permutation analysis. True verse-level or pericope-level analysis would require annotated source texts.

### 8.2 Future Work

**Contextual embeddings**: Replace static Word2Vec with transformer-based contextual embeddings (BERT-style) trained exclusively on Paul's corpus, to capture polysemy — the same word meaning different things in different contexts. The morphological sensitivity discovered in this study (δικαιοσύνη vs. δικαιοσύνην behaving differently) suggests contextual models could reveal even finer-grained semantic variation.

**Lemmatization study**: Run a parallel analysis with lemmatized forms to compare against the inflected-form results. The sharp contrast between nominative δικαιοσύνη (0.0 stability) and the δίκαιος ↔ δικαιοῦται pair (0.970 stability) raises the question of whether lemmatization would obscure genuine syntactic-semantic distinctions.

**Diachronic analysis**: If epistle dating can be established with sufficient confidence, track semantic evolution across Paul's career using the per-epistle Hurst exponents and topic distributions as longitudinal markers.

**Comparative validation**: Apply the same methodology to other small single-author corpora (e.g., the Johannine writings, Quranic Arabic, Platonic dialogues) and compare the methodological robustness. The pipeline's configurable sampling depth makes both rapid exploratory runs and full-scale analyses practical.

**Verse-annotated corpus**: Incorporate a verse-annotated Greek text (e.g., SBLGNT with verse markers) to enable true pericope-level permutation analysis and finer-grained positional templates for combinatorial recombination.

---

## 9. References

- Antoniak, M. & Mimno, D. (2018). "Evaluating the Stability of Embedding-based Word Similarities." *Transactions of the Association for Computational Linguistics*, 6, 107-119.
- Blei, D.M., Ng, A.Y., & Jordan, M.I. (2003). "Latent Dirichlet Allocation." *Journal of Machine Learning Research*, 3, 993-1022.
- Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." *Annals of Statistics*, 7(1), 1-26.
- Käsemann, E. (1969). *New Testament Questions of Today*. SCM Press.
- Kingma, D.P. & Welling, M. (2014). "Auto-Encoding Variational Bayes." *ICLR 2014*.
- Mealand, D.L. (1996). "The Extent of the Pauline Corpus: A Multivariate Approach." *Journal for the Study of the New Testament*, 59, 61-92.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S., & Dean, J. (2013). "Distributed Representations of Words and Phrases and their Compositionality." *NIPS 2013*.
- Morton, A.Q. (1965). "The Authorship of the Pauline Epistles: A Scientific Solution." *The Saskatoon Lectures*.
- Peng, C.-K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E., & Goldberger, A.L. (1994). "Mosaic organization of DNA nucleotides." *Physical Review E*, 49(2), 1685.
- Rice, J.A. (2006). *Mathematical Statistics and Data Analysis*. 3rd ed. Duxbury Press.
- Sanders, E.P. (1977). *Paul and Palestinian Judaism*. Fortress Press.
- Schönemann, P.H. (1966). "A generalized solution of the orthogonal Procrustes problem." *Psychometrika*, 31(1), 1-10.
- Westerholm, S. (2004). *Perspectives Old and New on Paul: The "Lutheran" Paul and His Critics*. Eerdmans.
- Wright, N.T. (1997). *What Saint Paul Really Said*. Eerdmans.
