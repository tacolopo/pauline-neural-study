"""
Tests for the Pauline Neural Study pipeline.

These tests use a minimal synthetic corpus to validate the pipeline
components without requiring network access or large computation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from pauline.corpus.loader import PaulineCorpus, Epistle, Verse, PAULINE_EPISTLES
from pauline.bootstrap.sampler import BootstrapSampler, SamplingLevel
from pauline.combinatorial.recombiner import CombinatorialRecombiner


def _make_test_corpus() -> PaulineCorpus:
    """Create a minimal test corpus with synthetic Pauline-style text."""
    epistles = []

    # Mini Romans
    romans_verses = [
        Verse("Romans", 1, 1, "Paul a servant of Jesus Christ called to be an apostle separated unto the gospel of God."),
        Verse("Romans", 1, 2, "Which he had promised before by his prophets in the holy scriptures."),
        Verse("Romans", 1, 3, "Concerning his Son Jesus Christ our Lord who was made of the seed of David according to the flesh."),
        Verse("Romans", 1, 4, "And declared to be the Son of God with power according to the spirit of holiness by the resurrection from the dead."),
        Verse("Romans", 1, 5, "By whom we have received grace and apostleship for obedience to the faith among all nations for his name."),
        Verse("Romans", 3, 21, "But now the righteousness of God without the law is manifested being witnessed by the law and the prophets."),
        Verse("Romans", 3, 22, "Even the righteousness of God which is by faith of Jesus Christ unto all and upon all them that believe for there is no difference."),
        Verse("Romans", 3, 23, "For all have sinned and come short of the glory of God."),
        Verse("Romans", 3, 24, "Being justified freely by his grace through the redemption that is in Christ Jesus."),
        Verse("Romans", 3, 25, "Whom God has set forth to be a propitiation through faith in his blood to declare his righteousness."),
        Verse("Romans", 5, 1, "Therefore being justified by faith we have peace with God through our Lord Jesus Christ."),
        Verse("Romans", 5, 2, "By whom also we have access by faith into this grace wherein we stand and rejoice in hope of the glory of God."),
        Verse("Romans", 8, 1, "There is therefore now no condemnation to them which are in Christ Jesus who walk not after the flesh but after the spirit."),
        Verse("Romans", 8, 2, "For the law of the spirit of life in Christ Jesus has made me free from the law of sin and death."),
    ]
    epistles.append(Epistle(name="Romans", verses=romans_verses))

    # Mini 1 Corinthians
    cor1_verses = [
        Verse("1 Corinthians", 1, 1, "Paul called to be an apostle of Jesus Christ through the will of God."),
        Verse("1 Corinthians", 1, 2, "Unto the church of God which is at Corinth to them that are sanctified in Christ Jesus called to be saints."),
        Verse("1 Corinthians", 1, 3, "Grace be unto you and peace from God our Father and from the Lord Jesus Christ."),
        Verse("1 Corinthians", 1, 18, "For the preaching of the cross is to them that perish foolishness but unto us which are saved it is the power of God."),
        Verse("1 Corinthians", 1, 30, "But of him are ye in Christ Jesus who of God is made unto us wisdom and righteousness and sanctification and redemption."),
        Verse("1 Corinthians", 13, 13, "And now abides faith hope love these three but the greatest of these is love."),
        Verse("1 Corinthians", 15, 3, "For I delivered unto you first of all that which I also received how that Christ died for our sins according to the scriptures."),
        Verse("1 Corinthians", 15, 4, "And that he was buried and that he rose again the third day according to the scriptures."),
    ]
    epistles.append(Epistle(name="1 Corinthians", verses=cor1_verses))

    # Mini Galatians
    gal_verses = [
        Verse("Galatians", 1, 1, "Paul an apostle not of men neither by man but by Jesus Christ and God the Father who raised him from the dead."),
        Verse("Galatians", 2, 16, "Knowing that a man is not justified by the works of the law but by the faith of Jesus Christ."),
        Verse("Galatians", 2, 20, "I am crucified with Christ nevertheless I live yet not I but Christ lives in me and the life which I now live in the flesh I live by the faith of the Son of God who loved me and gave himself for me."),
        Verse("Galatians", 3, 11, "But that no man is justified by the law in the sight of God it is evident for the just shall live by faith."),
        Verse("Galatians", 5, 1, "Stand fast therefore in the liberty wherewith Christ has made us free and be not entangled again with the yoke of bondage."),
        Verse("Galatians", 5, 22, "But the fruit of the spirit is love joy peace longsuffering gentleness goodness faith."),
    ]
    epistles.append(Epistle(name="Galatians", verses=gal_verses))

    # Mini Philippians
    phil_verses = [
        Verse("Philippians", 1, 1, "Paul and Timothy servants of Jesus Christ to all the saints in Christ Jesus which are at Philippi."),
        Verse("Philippians", 2, 5, "Let this mind be in you which was also in Christ Jesus."),
        Verse("Philippians", 3, 9, "And be found in him not having mine own righteousness which is of the law but that which is through the faith of Christ the righteousness which is of God by faith."),
        Verse("Philippians", 4, 13, "I can do all things through Christ which strengthens me."),
    ]
    epistles.append(Epistle(name="Philippians", verses=phil_verses))

    # Mini 1 Thessalonians
    thess_verses = [
        Verse("1 Thessalonians", 1, 1, "Paul and Silvanus and Timothy unto the church of the Thessalonians which is in God the Father and in the Lord Jesus Christ grace be unto you and peace."),
        Verse("1 Thessalonians", 4, 14, "For if we believe that Jesus died and rose again even so them also which sleep in Jesus will God bring with him."),
        Verse("1 Thessalonians", 5, 8, "But let us who are of the day be sober putting on the breastplate of faith and love and for a helmet the hope of salvation."),
    ]
    epistles.append(Epistle(name="1 Thessalonians", verses=thess_verses))

    # Mini 2 Corinthians
    cor2_verses = [
        Verse("2 Corinthians", 1, 1, "Paul an apostle of Jesus Christ by the will of God and Timothy our brother unto the church of God which is at Corinth."),
        Verse("2 Corinthians", 5, 17, "Therefore if any man be in Christ he is a new creature old things are passed away behold all things are become new."),
        Verse("2 Corinthians", 5, 21, "For he has made him to be sin for us who knew no sin that we might be made the righteousness of God in him."),
    ]
    epistles.append(Epistle(name="2 Corinthians", verses=cor2_verses))

    # Mini Philemon
    phlm_verses = [
        Verse("Philemon", 1, 1, "Paul a prisoner of Jesus Christ and Timothy our brother unto Philemon our dearly beloved."),
        Verse("Philemon", 1, 3, "Grace to you and peace from God our Father and the Lord Jesus Christ."),
        Verse("Philemon", 1, 6, "That the communication of thy faith may become effectual by the acknowledging of every good thing which is in you in Christ Jesus."),
    ]
    epistles.append(Epistle(name="Philemon", verses=phlm_verses))

    return PaulineCorpus(epistles=epistles, undisputed_only=True)


class TestCorpus:
    """Test corpus loading and basic properties."""

    def test_corpus_loads(self):
        corpus = _make_test_corpus()
        assert len(corpus.epistles) == 7

    def test_epistle_names(self):
        corpus = _make_test_corpus()
        names = corpus.epistle_names
        assert "Romans" in names
        assert "Galatians" in names

    def test_vocabulary(self):
        corpus = _make_test_corpus()
        vocab = corpus.vocabulary
        assert "faith" in vocab
        assert "christ" in vocab
        assert "god" in vocab

    def test_word_count(self):
        corpus = _make_test_corpus()
        assert corpus.total_words > 100

    def test_sentences(self):
        corpus = _make_test_corpus()
        assert corpus.total_sentences > 10

    def test_word_frequency(self):
        corpus = _make_test_corpus()
        freq = corpus.word_frequency()
        # Common Pauline words should appear frequently
        assert freq.get("christ", 0) > 0
        assert freq.get("god", 0) > 0

    def test_summary(self):
        corpus = _make_test_corpus()
        summary = corpus.summary()
        assert summary["epistles"] == 7
        assert summary["total_words"] > 0
        assert summary["vocabulary_size"] > 0


class TestBootstrap:
    """Test bootstrap sampling."""

    def test_sentence_level_bootstrap(self):
        corpus = _make_test_corpus()
        sampler = BootstrapSampler(corpus, seed=42)
        result = sampler.sample(n_samples=10, level=SamplingLevel.SENTENCE)
        assert result.n_samples == 10
        assert len(result.samples) == 10

    def test_epistle_level_bootstrap(self):
        corpus = _make_test_corpus()
        sampler = BootstrapSampler(corpus, seed=42)
        result = sampler.sample(n_samples=10, level=SamplingLevel.EPISTLE)
        assert result.n_samples == 10

    def test_chapter_level_bootstrap(self):
        corpus = _make_test_corpus()
        sampler = BootstrapSampler(corpus, seed=42)
        result = sampler.sample(n_samples=10, level=SamplingLevel.CHAPTER)
        assert result.n_samples == 10

    def test_bootstrap_produces_words(self):
        corpus = _make_test_corpus()
        sampler = BootstrapSampler(corpus, seed=42)
        result = sampler.sample(n_samples=5, level=SamplingLevel.SENTENCE)
        assert result.total_words_generated > 0
        assert result.avg_sample_size > 0

    def test_jackknife(self):
        corpus = _make_test_corpus()
        sampler = BootstrapSampler(corpus, seed=42)
        jk = sampler.jackknife_epistles()
        assert len(jk) == 7  # One for each undisputed epistle

    def test_multi_level_sample(self):
        corpus = _make_test_corpus()
        sampler = BootstrapSampler(corpus, seed=42)
        results = sampler.multi_level_sample(
            n_samples=5,
            levels=[SamplingLevel.SENTENCE, SamplingLevel.CHAPTER],
        )
        assert SamplingLevel.SENTENCE in results
        assert SamplingLevel.CHAPTER in results

    def test_reproducibility(self):
        corpus = _make_test_corpus()
        s1 = BootstrapSampler(corpus, seed=42)
        s2 = BootstrapSampler(corpus, seed=42)
        r1 = s1.sample(n_samples=5, level=SamplingLevel.SENTENCE)
        r2 = s2.sample(n_samples=5, level=SamplingLevel.SENTENCE)
        # Same seed should produce same samples
        assert r1.samples[0].source_indices == r2.samples[0].source_indices


class TestCombinatorial:
    """Test combinatorial recombination."""

    def test_prepare(self):
        corpus = _make_test_corpus()
        recombiner = CombinatorialRecombiner(corpus, seed=42)
        recombiner.prepare()
        assert recombiner._prepared is True
        assert len(recombiner._templates) > 0

    def test_random_generation(self):
        corpus = _make_test_corpus()
        recombiner = CombinatorialRecombiner(corpus, seed=42)
        result = recombiner.generate_random(n_sentences=10, min_length=3, max_length=50)
        assert len(result.generated_sentences) == 10
        assert result.generation_method == "random_slot_filling"

    def test_constrained_generation(self):
        corpus = _make_test_corpus()
        recombiner = CombinatorialRecombiner(corpus, seed=42)
        result = recombiner.generate_constrained(
            n_sentences=10, bigram_threshold=0, min_length=3, max_length=50
        )
        assert len(result.generated_sentences) == 10
        assert "pauline_vocabulary_only" in result.constraints_applied

    def test_vocabulary_constraint(self):
        """Verify all generated words come from Paul's vocabulary."""
        corpus = _make_test_corpus()
        paul_vocab = corpus.vocabulary
        recombiner = CombinatorialRecombiner(corpus, seed=42)
        result = recombiner.generate_random(n_sentences=20, min_length=3, max_length=50)

        for sent in result.generated_sentences:
            words = sent.lower().split()
            for word in words:
                if word.isalpha():
                    assert word in paul_vocab or word in {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "for", "and", "but", "or", "not", "no", "by", "with", "from", "that", "which", "who", "whom", "this", "these", "those", "it", "he", "him", "his", "they", "them", "their", "we", "us", "our", "you", "your", "i", "me", "my"}, \
                        f"Word '{word}' not in Paul's vocabulary"


class TestCorpusJSON:
    """Test corpus JSON serialization."""

    def test_save_and_load(self):
        corpus = _make_test_corpus()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            import json
            data = {
                "epistles": [
                    {
                        "name": ep.name,
                        "verses": [
                            {"chapter": v.chapter, "verse": v.verse, "text": v.text}
                            for v in ep.verses
                        ],
                    }
                    for ep in corpus.epistles
                ]
            }
            path = Path(tmpdir) / "test_corpus.json"
            with open(path, "w") as f:
                json.dump(data, f)

            # Load
            loaded = PaulineCorpus.from_json(path, undisputed_only=True)
            assert len(loaded.epistles) == 7
            assert loaded.total_words > 0
