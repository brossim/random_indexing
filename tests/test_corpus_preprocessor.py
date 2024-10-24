# Author        : Simon Bross
# Date          : March 10, 2023
# Python version: 3.9
# File purpose  : Unit tests for CorpusPreprocessor class


import string
from nltk.corpus import stopwords
from ..random_indexing import CorpusPreprocessor as Cp


class TestCorpusPreprocessor:

    # get stopwords and punctuation marks to be removed
    # (same as in CorpusPreprocessor)
    punctuation = string.punctuation.replace("", " ").split()
    nltk_stopwords = stopwords.words('english')
    all_stops = set(nltk_stopwords + punctuation)
    # tagset list from https://www.nltk.org/book/ch05.html
    tagset = ["ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN",
              "NUM", "PRT", "PRON", "VERB", ".", "X"]

    def test_is_tagged(self):
        corpus = [
            ["This", "is", "a", "test", "corpus"]
        ]
        corpus = Cp.CorpusPreprocessor(corpus=corpus).preprocess_corpus()
        # tokens in corpus look like: apple_NOUN
        for token in corpus:
            assert token.split("_")[1] in self.tagset

    def test_no_stopwords(self):
        corpus = [
             ["This", "is", "a", "test", "corpus", "!"]
        ]
        corpus = Cp.CorpusPreprocessor(corpus=corpus).preprocess_corpus()
        for stop in self.all_stops:
            assert stop not in corpus

    def test_no_x_postag(self):
        # there should not be tokens tagged with 'X', i.e. 'other'
        # in universal tagset
        corpus = [
            ["This", "is", "a", "nonsensical", "word", "rfjghkdjgksd",
             "fail§%§"]
        ]
        corpus = Cp.CorpusPreprocessor(corpus=corpus).preprocess_corpus()
        for token in corpus:
            assert "X" not in token.split("_")[1]

    def test_no_uppercase(self):
        corpus = [
            ["This", "Is", "An", "Uppercased", "Corpus"]
        ]
        corpus = Cp.CorpusPreprocessor(corpus=corpus).preprocess_corpus()
        for token in corpus:
            assert not token.isupper()

    def test_no_possessive_s(self):
        # woman's should be woman_NOUN in processed corpus
        corpus = [
            ["The", "woman's", "plan", "failed"]
        ]
        corpus = Cp.CorpusPreprocessor(corpus=corpus).preprocess_corpus()
        assert "woman_NOUN" in corpus

    def test_no_integer(self):
        corpus = [
            ["There", "are", "3", "dogs", "in", "the", "park"]
        ]
        corpus = Cp.CorpusPreprocessor(corpus=corpus).preprocess_corpus()
        without_pos = map(lambda x: x.split("_")[1], corpus)
        assert "3" not in without_pos and "3.7" not in without_pos

    def test_no_float(self):
        corpus = [
            ["There", "are", "about", "83.2", "million", "people", "living",
             "in", "Germany"]
        ]
        corpus = Cp.CorpusPreprocessor(corpus=corpus).preprocess_corpus()
        without_pos = map(lambda x: x.split("_")[1], corpus)
        assert "83.2" not in without_pos

    def test_is_lemmatized(self):
        corpus = [
            ["She", "sings", "two", "arias"]
        ]
        corpus = Cp.CorpusPreprocessor(corpus=corpus).preprocess_corpus()
        assert "sing_VERB" in corpus and "aria_NOUN" in corpus
