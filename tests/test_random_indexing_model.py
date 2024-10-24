# Author        : Simon Bross
# Date          : March 10, 2023
# Python version: 3.9
# File purpose  : Unit tests for RandomIndexingModel class


import os
from ..random_indexing import RandomIndexingModel as Rim
from ..random_indexing import CorpusPreprocessor as Cp


class TestRandomIndexingModel:
    # take first 10k words from brown preprocessed as corpus
    corpus = Cp.CorpusPreprocessor(corpus='brown').preprocess_corpus()[:10000]
    model = Rim.RandomIndexingModel(6, 6, 500, 50, corpus=corpus,
                                    name='brown_shortened')

    def test_str_repr(self):
        assert str(self.model) == "model_brown_shortened_6_6_500_50"

    def test_stored_correctly(self):
        path = os.path.join("..", "models", str(self.model) + ".pkl")
        assert os.path.exists(path)

    def test_word_vector_lengths(self):
        for vector in self.model.model.values():
            assert len(vector) == self.model.dim_size

    def test_model_completeness(self):
        # check if all corpus words appear in model
        for token in self.corpus:
            assert token in self.model.model

    def test_index_vector_lengths(self):
        for index_vec in self.model.index_vectors.values():
            assert len(index_vec) == self.model.dim_size

    def test_non_zero_dims(self):
        # assure 50 non-zero dims (i.e. either -1 or 1) are in
        # every index vector
        for index_vec in self.model.index_vectors.values():
            assert len(index_vec[index_vec != 0]) == 50
