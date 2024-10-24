# Author        : Simon Bross
# Date          : March 11, 2023
# Python version: 3.9
# File purpose  : Unit tests for BrownDocVec class


import sys
import os
# append path as BrownDocVec imports CorpusPreprocessor
# from the same folder. Running this file, the
# import statement "from CorpusPreprocessor import CorpusPreprocessor"
# in the BrownDocVec class cannot be executed
sys.path.append(os.path.join("..", "random_indexing"))  # noqa: E402
from ..random_indexing import BrownDocVec as Bdc
from ..random_indexing import RandomIndexingModel as Rim
from ..random_indexing import CorpusPreprocessor as Cp


class TestBrownDocVec:

    data_creator = Bdc.BrownDocVec()
    categories = data_creator.categories
    test_corpus = Cp.CorpusPreprocessor('brown').preprocess_corpus()[:1000]
    test_model = Rim.RandomIndexingModel(6, 6, 500, 50, test_corpus,
                                         "unit_test")
    data_creator.create_csv_data(test_model.model, str(test_model),
                                 mode="training")

    def test_every_category_in_split(self):
        # assert that every split contains at least a document from every class
        for cat in self.categories:
            assert cat in self.data_creator.y_train, f"'{cat}' not in y_train"
            assert cat in self.data_creator.y_test, f"'{cat}' not in y_test"
            assert cat in self.data_creator.y_val, f"'{cat}' not in y_val"
