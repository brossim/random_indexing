# Author        : Simon Bross
# Date          : March 6, 2023
# Python version: 3.9
# File purpose  : Implementation of IntrinsicEvaluator computing the spearman
#                 correlation between the RI word vector relation and human
#                 ratings using simlex/wordsim datasets

import os
import pandas as pd
from evaluate_document_vectors import cosine_similarity
from numpy import nan
from scipy.stats import spearmanr


class IntrinsicEvaluator:
    """
    IntrinsicEvaluator that can either load Simlex or Wordsim
    datasets preprocessed in order to be compatible with the
    corpus provided by the CorpusPreprocessor, i.e. words are
    extended by their respective POS tags (universal tagset).
    Datasets are loaded into dataframes containing the following
    columns: word1, word2, value (specific similarity/
    relatedness score). In order to evaluate the RI model, the
    spearman correlation is computed by determining the cosine
    similarity between word pairs (word1, word2) from the model,
    opposed to the word pair scores (value) from the datasets.
    Spearman correlation scores range from [-1, 1], where 1 and
    -1 are high (negative) correlations and values around 0
    do not account for a correlation (or account for only rather
    low correlations).
    """
    def __init__(self, path, model):
        """
        Creates an instance of IntrinsicEvaluator.
        @param path: Valid path leading to either simlex or wordsim
        data, as str. Depending on the string, either simlex or wordsim
        data is loaded into the dataset attribute.
        @param model: Random Indexing model as dict, in which words are
        mapped to their respective word vectors.
        """
        assert os.path.exists(path) and (
                "SimLex-999" in path or "wordsim" in path
        ), "Data not found under the given path"
        self.__path = path

        assert isinstance(model, dict), \
            "RI model must be a dictionary mapping words to their" \
            " context vectors"
        self.__model = model

        # load simlex data, should be "SimLex-999.txt"
        if "SimLex-999" in path:
            self.__dataset = self.__load_simlex()

        # load wordsim data
        # should be either "wordsim_relatedness_goldstandard.txt" or
        # wordsim_similarity_goldstandard.txt"
        elif "wordsim353" in path:
            self.__dataset = self.__load_wordsim()

    def __load_simlex(self):
        """
        Loads the simlex dataset into a pandas dataframe and preprocesses
        it to match the tokens in the corpus (in which words are extended by
        their POS tags). The POS tags from the dataset are converted into
        its equivalents used from universal tagset in corpus preprocessing.
        @return: Preprocessed wordsim dataset as pandas dataframe, subset
        to only include the word pairs (word1, word2) and the similarity score
        (value).
        """
        df = pd.read_table(self.__path, sep="\t", header=0)
        # make POS tags from simlex compatible with universal tagset
        # used in corpus preprocessing
        simlex_to_universal = {
            "N": "NOUN",
            "V": "VERB",
            "A": "ADJ"
        }
        df["POS"] = df["POS"].map(lambda pos: simlex_to_universal[pos])
        # expand words by its POS tags
        df["word1"] = df["word1"] + "_" + df["POS"]
        df["word2"] = df["word2"] + "_" + df["POS"]
        # make simlex and wordsim columns identical for compatibility
        # with compute_spearman method
        df.rename(columns={"SimLex999": "value"}, inplace=True)
        return df[["word1", "word2", "value"]]

    def __load_wordsim(self):
        """
        Loads the wordsim dataset into a pandas dataframe and preprocesses
        it to match not only the corpus (in which words are extended by
        their POS tags), but also to give it a structure (column names)
        that compute_spearman can handle from both simlex and wordsim data
        @return: Preprocessed wordsim dataset as pandas dataframe.
        """
        df = pd.read_table(self.__path, sep="\t",
                           names=["word1", "word2", "value"])
        # expand words by their POS tags for corpus compatibility
        # (corpus preprocessing expanded tokens by POS tags to distinguish
        # homographs)
        # some words are uppercased, corpus is lowercased
        df["word1"] = df["word1"].map(lambda word: word.lower()) + "_NOUN"
        df["word2"] = df["word2"].map(lambda word: word.lower()) + "_NOUN"
        return df

    def compute_spearman(self):
        """
        Computes the spearman correlation between human ratings of
        word pairs and the cosine similarity of word pairs from a
        random indexing model. Iterating over the word pairs in the
        given dataset, the cosine similarity is only computed if and
        only if the RI model contains both words. Otherwise, "nan" is
        used instead, thus ensuring both that the lengths of human
        ratings and cosine similarity scores equal and that the order
        is maintained. If nan values exist, the calculation is performed
        ignoring them.
        @return: instance of SignificanceResult
        """
        human_ratings = self.__dataset["value"].tolist()
        word_pairs = [
            (word1, word2) for word1, word2
            in zip(
                self.__dataset["word1"].tolist(),
                self.__dataset["word2"].tolist()
            )
        ]
        # collect cosine similarity scores from word pairs
        ri_cos_values = []
        for word1, word2 in word_pairs:
            if word1 in self.__model and word2 in self.__model:
                vec1 = self.__model[word1]
                vec2 = self.__model[word2]
                assert len(vec1) == len(vec2)
                ri_cos_values.append(cosine_similarity(vec1, vec2))
            # if either word not in RI model, append nan as cosine similarity
            # cannot be computed
            else:
                ri_cos_values.append(nan)

        assert len(human_ratings) == len(ri_cos_values)
        result = spearmanr(human_ratings, ri_cos_values, nan_policy='omit')
        return result


if __name__ == '__main__':
    # example usage
    # path1 must exist in order to run this
    from CorpusPreprocessor import CorpusPreprocessor
    from RandomIndexingModel import RandomIndexingModel
    cp = CorpusPreprocessor(corpus='brown')
    corpus = cp.preprocess_corpus()
    model1 = RandomIndexingModel(7, 7, 500, 50, corpus=corpus,
                                 name='brown')
    path1 = os.path.join("..", "data", "wordsim353_sim_rel",
                         "wordsim_relatedness_goldstandard.txt")
    intr_eval = IntrinsicEvaluator(path1, model1.model)
    print(intr_eval.compute_spearman())
