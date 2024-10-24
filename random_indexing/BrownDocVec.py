# Author        : Simon Bross
# Date          : March 9, 2023
# Python version: 3.9
# File purpose  : Implements part of the extrinsic evaluation of RIM models,
#                 i.e. for the category classification in the brown corpus.
#                 Prepares the data needed for the classifier.


import os
import numpy as np
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from colorama import Fore
from CorpusPreprocessor import CorpusPreprocessor


class BrownDocVec:
    """
    Creates the data for the extrinsic evaluation (document classification) of
    random indexing models. Features (X) are document vectors calculated as the
    sum of all word vectors present in the document, using a given RI model.
    Labels are the respective document categories from the brown corpus -
    c.f. https://www.nltk.org/book/ch02.html (1.3). The data is split into a
    training set (~70%) used to create the document vectors from different RI
    models, a validation set (~10%) used to test the different models, and a
    testing set (~20%) used to test the model that performed best on the
    validation set.
    """

    def __init__(self):
        # get all text categories from brown corpus, equals label set
        self.__categories = brown.categories()

        # map every category to its fileid
        # every category has more than 3 texts
        # --> every category will appear in test/training/validation split
        self.__cat_to_texts = {
            cat: brown.fileids(cat) for cat in self.__categories
        }

        # create datapoints, i.e. initially fileids that are later converted
        # into document vectors
        self.__X = [
            fileid for value in self.__cat_to_texts.values()
            for fileid in value
        ]

        # determine label for every datapoint in X
        self.__y = [
            cat for cat in self.__cat_to_texts
            for _ in range(len(self.__cat_to_texts[cat]))
        ]

        # make training (~70%), testing (~20%), and validation (~10%) split
        # first split call: make 80% training (adjusted in second call)
        # and 20% test (fixed)
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = \
            train_test_split(self.__X, self.__y, test_size=0.2,
                             random_state=123, stratify=self.__y)

        # second split call: adjust training 80% --> ~70% and create 10%
        # validation split, i.e. use train_size of 7/8 (= 87.5%)
        # However, this would exclude the class science_fiction as there
        # are only 6 texts pertaining to it; use 86% instead
        self.__X_train, self.__X_val, self.__y_train, self.__y_val = \
            train_test_split(self.__X_train, self.__y_train, train_size=0.86,
                             random_state=123, stratify=self.__y_train)

    # Getters are primarily used for unit test
    @property
    def categories(self):
        """
        Returns the list of categories, i.e. the entire label set.
        @return: List of strings.
        """
        return self.__categories

    @property
    def y_train(self):
        """
        Returns the labels for the training set.
        @return: List of strings.
        """
        return self.__y_train

    @property
    def y_val(self):
        """
        Returns the labels for the validation set.
        @return: List of strings.
        """
        return self.__y_val

    @property
    def y_test(self):
        """
        Returns the labels for the testing set.
        @return: List of strings.
        """
        return self.__y_test

    def create_csv_data(self, model, csv_name, mode=None):
        """
        Creates either the training, validation, or testing data given
        an RI model, i.e. the document vectors (X) and its labels (y).
        X and y are both stored as a comma separated .csv file.
        Files are stored in the 'extrinsic_eval' directory.
        @param csv_name: Name of the csv file that determines the file
        path (../extrinsic_eval/mode_csv_name + .csv/_labels.csv).
        Terminal interface in train_text_classification.py automatically
        provides the model name as csv_name.
        @param model: RIM model as dict.
        @param mode: Mode determining which data to be created. Must be
        either 'training', 'validation' or 'testing', as str.
        """
        assert mode in ['training', 'validation', 'testing'], \
            "Wrong mode provided. Either 'training' to create the data for " \
            "training (training set of document representations + labels), " \
            "'validation' (validation set of document representations + gold" \
            " labels to test the RI models), 'testing' (test set of document" \
            " representations + gold labels for testing the model that " \
            "performed best on the validation set)"
        # store document vectors computed using the given model
        doc_vecs = []
        # create document vector for every fileid in respective split
        if mode == 'training':
            X = self.__X_train
            y = self.__y_train
        elif mode == 'validation':
            X = self.__X_val
            y = self.__y_val
        else:
            X = self.__X_test
            y = self.__y_test

        for fileid in X:
            file_sents = brown.tagged_sents(fileids=fileid, tagset='universal')
            # preprocess every file
            cp = CorpusPreprocessor(corpus=file_sents)
            preprocessed = cp.preprocess_corpus()
            # get all word vectors from model that appear in the document
            word_vectors = [
                model[word] for word in preprocessed if word in model
            ]
            doc_vec = np.sum(word_vectors, axis=0)
            doc_vecs.append(doc_vec)
        # store document vectors as .csv in extrinsic_eval directory
        dir_path = os.path.join("..", "extrinsic_eval")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        x_path = os.path.join(
            dir_path, f"{mode}_{csv_name}.csv"
        )
        np.savetxt(x_path, doc_vecs, delimiter=',', newline="\n")
        label_path = os.path.join(
            dir_path, f"{mode}_{csv_name}_labels.csv")
        np.savetxt(label_path, y, delimiter=',', newline="\n",
                   fmt='%s')
        print(Fore.RED + f"{mode.capitalize()} document vectors successfully"
                         f" created and stored in {x_path}" + Fore.RESET)
        print(Fore.RED + f"{mode.capitalize()} labels successfully created and"
                         f" stored in {label_path}" + Fore.RESET)


if __name__ == '__main__':
    # example usage
    import time
    from RandomIndexingModel import RandomIndexingModel
    start = time.time()
    corpus = CorpusPreprocessor(corpus='brown').preprocess_corpus()
    test = BrownDocVec()
    test_model = RandomIndexingModel(6, 6, 500, 50, corpus=corpus,
                                     name='brown_example')
    test.create_csv_data(test_model.model, str(test_model), mode='training')
    print(time.time() - start, "s")
    # 24.8s for entire brown corpus, including corpus preprocessing and model
    # creation
