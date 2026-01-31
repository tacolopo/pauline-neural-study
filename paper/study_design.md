# Closed-Corpus Neural Semantics: A Novel Computational Approach to Pauline Theology Using Bootstrap Resampling and Combinatorial Expansion

## Abstract

We present a novel computational methodology for analyzing the semantic structure of the Pauline corpus — the letters attributed to the apostle Paul in the New Testament — using neural word embeddings trained exclusively on Paul's own words. The central challenge in Pauline studies is that contested theological terms such as *justification* (δικαίωσις), *faith* (πίστις), and *law* (νόμος) carry meanings that remain debated after two millennia of scholarship. While neural language models excel at discovering hidden semantic relationships, Paul's corpus (~32,000 words across 7 undisputed epistles) is far too small for standard training approaches. Critically, we argue that supplementing Paul's corpus with external texts — even contemporary Koine Greek sources — introduces undetectable semantic contamination, since theological interpretation lacks an objective validation metric.

Our solution treats Paul's corpus as a **closed statistical universe** and applies mathematically rigorous expansion techniques operating exclusively on Paul's vocabulary, syntax, and co-occurrence patterns. Drawing on an analogy to the Central Limit Theorem, we employ multi-level bootstrap resampling to generate thousands of valid "alternate Pauline corpora," train word embeddings on each, and aggregate results to identify stable semantic relationships. We complement this with combinatorial recombination within Paul's grammatical structures, cross-epistle transfer analysis, fractal self-similarity measurement, permutation language modeling, variational autoencoders, and hierarchical Bayesian topic modeling — all constrained to Paul's closed vocabulary.

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

Neural word embedding models typically require millions of words to produce reliable results. Google's original Word2Vec was trained on 1.6 billion words [Mikolov et al., 2013]. Even "small" NLP datasets usually contain hundreds of thousands of words. Paul's undisputed corpus contains approximately 32,000 words — roughly three to four orders of magnitude smaller than standard requirements.

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

**Epistle-level** ($n = 7$ units): Each bootstrap sample draws 7 epistles with replacement from the 7 undisputed letters. This preserves macro-theological structure but allows some epistles to appear multiple times while others are absent. With 7 units, there are $7^7 = 823,543$ possible bootstrap samples.

**Chapter-level** ($n \approx 60$ units): Each sample draws chapters with replacement. This preserves within-chapter coherence while allowing different chapter combinations.

**Sentence-level** ($n \approx 1,200$ units): Each sample draws individual sentences with replacement. This maximizes combinatorial diversity while preserving sentence-level syntax and meaning.

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

**Leave-one-out**: For each epistle $E_i$, train embeddings on the remaining 6 epistles. Compare word relationships to the full-corpus baseline. The difference reveals the influence of $E_i$ on specific semantic relationships.

**All-subsets**: Analyze all $\binom{7}{3} + \binom{7}{4} + \ldots + \binom{7}{7} = 99$ subsets of 3 or more epistles. This provides the most complete picture of semantic stability.

Results are classified into three categories:

1. **Stable relationships**: Consistent across all epistle subsets → genuine Pauline semantics
2. **Variable relationships**: Change significantly depending on included epistles → context-dependent usage
3. **Epistle-specific relationships**: Dominated by one particular epistle → situational theology

### 4.4 Phase 4: Combinatorial Recombination

We expand the corpus by generating synthetic Pauline sentences using only Paul's own grammatical structures and vocabulary:

1. **Template extraction**: Parse all Pauline sentences into POS-tag templates using NLTK's averaged perceptron tagger.
2. **Vocabulary-by-slot mapping**: For each POS tag, collect all words Paul used in that grammatical role.
3. **Constrained generation**: Fill templates with Paul's words, preferring word pairs that Paul actually used together (co-occurrence constraint).

Every generated sentence is guaranteed to use only Paul's vocabulary in Paul's syntactic structures. This is not text generation in the LLM sense — it is a controlled combinatorial expansion that creates valid training samples while maintaining Pauline authenticity.

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
| POS tagging | NLTK | Syntactic template extraction |
| Deep learning | PyTorch 2.0+ | VAE encoder/decoder |
| Bayesian inference | NumPy | Collapsed Gibbs sampling |
| Visualization | Matplotlib, Seaborn | Embedding projections, heatmaps |
| Dimensionality reduction | scikit-learn, UMAP | t-SNE and UMAP projections |

The pipeline is modular: each phase can be run independently. Configuration is managed via YAML files. The corpus is fetched from public domain sources (World English Bible via bible-api.com) and cached locally as JSON.

For the full-scale analysis, we provide execution scripts for Google Cloud Platform VMs. Recommended specs: 4 vCPUs, 16 GB RAM for CPU phases; add NVIDIA T4 GPU for VAE training.

---

## 6. Expected Contributions

### 6.1 Methodological Contributions

1. **Closed-corpus neural semantics**: A new paradigm for computational analysis of small textual corpora where external augmentation is theoretically invalid.

2. **Bootstrap stability as primary metric**: Using embedding stability across bootstrap samples not merely as a quality check but as the primary indicator of genuine vs. artifactual semantic relationships.

3. **Multi-level granularity analysis**: Systematic comparison of semantic stability at epistle, chapter, and sentence levels, revealing which relationships are robust across all scales.

### 6.2 Pauline Studies Contributions

4. **Empirical semantic maps**: Data-driven maps of Paul's semantic field structure, identifying which theological terms genuinely cluster together in Paul's usage (not in later interpretive traditions).

5. **Epistle-specific theology**: Quantitative identification of which semantic relationships are universal to Paul vs. specific to individual epistles, informing the debate about development in Paul's thought.

6. **Contested term analysis**: New evidence for the semantic range of terms like δικαιοσύνη, πίστις, and νόμος, grounded exclusively in Paul's own co-occurrence patterns.

### 6.3 Digital Humanities Contributions

7. **Reproducible framework**: An open-source, configurable pipeline that can be applied to any small author-specific corpus (e.g., Plato's dialogues, Shakespeare's sonnets, Emily Dickinson's poems).

---

## 7. Limitations and Future Work

### 7.1 Limitations

**Translation dependency**: When using English text, results reflect the translator's interpretation as well as Paul's original meaning. Greek-language analysis is preferred but requires specialized tokenization.

**Embedding limitations**: Word2Vec captures distributional semantics but cannot directly model syntactic relationships, discourse structure, or pragmatic meaning. Paul's rhetorical strategies may not be fully captured.

**Bootstrap assumptions**: The CLT analogy assumes that bootstrap samples are sufficiently representative of the underlying distribution. With only 7 epistles (the smallest sampling unit), epistle-level bootstrap has limited statistical power.

**Combinatorial authenticity**: Generated sentences, while using only Paul's vocabulary and structures, are not Paul's actual words. Their value is as training data for embedding models, not as theological evidence.

### 7.2 Future Work

**Greek text analysis**: Apply the full pipeline to Koine Greek text (e.g., SBLGNT, Westcott-Hort) for maximal scholarly precision.

**Contextual embeddings**: Replace static Word2Vec with transformer-based contextual embeddings (BERT-style) trained exclusively on Paul's corpus, to capture polysemy — the same word meaning different things in different contexts.

**Diachronic analysis**: If epistle dating can be established with sufficient confidence, track semantic evolution across Paul's career.

**Comparative validation**: Apply the same methodology to other small single-author corpora (e.g., the Johannine writings, the Pastoral Epistles) and compare the methodological robustness.

---

## 8. References

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
