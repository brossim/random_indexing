# Author        : Simon Bross
# Date          : February 16, 2023
# Python version: 3.9
# File purpose  : Implementation of a word space model using random indexing

import pickle
import random
import os
import numpy as np
from numpy.random import shuffle
from colorama import Fore


class RandomIndexingModel:

    def __init__(self, left_context, right_context, dim_size, non_zero_dims,
                 corpus, name=None):
        """
        Creates a Random Indexing Model that is computed using a (preprocessed)
        corpus in which a context window (of differing sizes depending on the
        given left and right context size) is used to traverse the corpus,
        thereby dding up the initially ternary and randomly distributed index
        vectors of words that co-occur together within the window. The model
        is stored as a .pkl file to be reused/loaded for evaluation.
        @param left_context: Number determining the left context window size,
        as int greater than 0.
        @param right_context: Number determining the right context window size,
        as int greater than 0.
        @param dim_size: Vector dimension as int, greater than or equal to 500.
        @param non_zero_dims: Number of dimensions to be non-zero,
        i.e. in [1, -1], as int.
        @param corpus: (Preprocessed) corpus on which the model is trained on,
        a list of strings (tokens).
        @param name: Optional name (e.g. 'brown') used for determining the
        model's string representation and file name.
        """
        assert isinstance(left_context, int) and left_context > 0, \
            "Left window context size must be an integer greater than 0"
        self.__left_context = left_context

        assert isinstance(right_context, int) and right_context > 0, \
            "Right window context size must be an integer greater than 0"
        self.__right_context = right_context

        assert isinstance(dim_size, int) and dim_size >= 500, \
            "Vector dimension must be an integer greater than or equal to 500"
        self.__dim_size = dim_size

        assert isinstance(non_zero_dims, int), \
            "The number of non-zero vector values must be an integer"
        assert non_zero_dims <= dim_size, \
            "Non zero dimensions cannot be greater than the " \
            "total number of dimensions"
        self.__non_zero_dims = non_zero_dims

        assert isinstance(corpus, list), "Corpus must be a list of strings"
        for ele in list(set(corpus)):
            assert isinstance(ele, str), \
                "Every element in the corpus must be a string"

        if name is not None:
            assert isinstance(name, str) and len(name) >= 3, \
                "Name parameter must be a string of at least three characters."
            self.name = name

        self.__corpus = corpus
        self.__index_vectors = {}
        self.__model = {}
        self.__create_index_vectors()
        self.__train_model()
        self.__store_model()

    def __str__(self):
        """
        Determines the string representation of a RandomIndexingModel object,
        depending on whether the object was instantiated with a name.
        @return: Str.
        """
        left = self.left_context
        right = self.right_context
        dims = self.dim_size
        non_zero = self.non_zero_dims
        if hasattr(self, "name"):
            return f"model_{self.name}_{left}_{right}_{dims}_{non_zero}"
        else:
            return f"model_{left}_{right}_{dims}_{non_zero}"

    @property
    def corpus(self):
        """
        Returns the corpus that the model was trained on.
        @return: Corpus as list of strings (tokens).
        """
        return self.__corpus

    @property
    def model(self):
        """
        Returns trained random indexing model.
        @return: Model as dict.
        """
        return self.__model

    @property
    def dim_size(self):
        """
        Returns the total vector dimension.
        @return: Dimension as int.
        """
        return self.__dim_size

    @property
    def non_zero_dims(self):
        """
        Returns the number of non-zero dimensions.
        @return: Dimensions as int.
        """
        return self.__non_zero_dims

    @property
    def left_context(self):
        """
        Returns the left context window size.
        @return: Left context as int.
        """
        return self.__left_context

    @property
    def right_context(self):
        """
        Returns the right context window size.
        @return: Right context as int.
        """
        return self.__right_context

    @property
    def index_vectors(self):
        """
        Returns the dictionary containing the index vector of every
        corpus type.
        @return: Dict.
        """
        return self.__index_vectors

    def __create_index_vectors(self):
        """
        Creates a ternary index vector of size dim_size with a random
        distribution of values in [1, -1] for every type in the corpus,
        depending on the given number of values to be non-zero.
        The remaining dimensions are filled with zeros.
        """
        # determine total number of dimensions to be zero
        num_zeros = self.dim_size - self.non_zero_dims
        # -1 and 1 have the same probability to be chosen
        non_zero_choice = np.array([-1, 1])
        for token in set(self.corpus):
            plain_vector = np.array(
                [0] * num_zeros
                + [random.choice(non_zero_choice)
                   for _ in range(self.non_zero_dims)]
            )
            # make distribution random
            shuffle(plain_vector)
            self.index_vectors[token] = plain_vector

    def __train_model(self):
        """
        Computes the context vectors for the model by passing through the
        corpus using context windows, thereby adding up the word vectors
        of the words that co-occur together within the window. However,
        words at the same index position cannot co-occur with themselves.
        """
        left = self.left_context
        right = self.right_context
        for idx, token in enumerate(self.corpus):
            # prepare window for every token
            window = self.corpus[
                     max(0, idx - left):
                     min(len(self.corpus) + 1, idx + right + 1)
                     ]
            # determine specific token index in window
            # consider that the token might occur more than once
            # in the window and that list.index(token) might therefore
            # yield an incorrect result
            # case 1: left window context size is confined
            # at the beginning of the corpus list, i.e. given context
            # size was not applicable and had to be adjusted
            if max(0, idx - left) < left:
                token_window_idx = idx
            else:
                # case 2 (general) / case 3: left context is not
                # confined/right context is confined
                # if no context is confined or if the right context is
                # confined, the index position of the token in question
                # equals the left context size (zero-indexed)
                token_window_idx = left

            # compute context vectors
            # get all relevant index vectors in context
            # token cannot co-occur with itself at the same index position
            index_vecs = [
                self.index_vectors[context_token] for context_token in window
                if window.index(context_token) != token_window_idx
            ]
            context_sum = np.sum(index_vecs, axis=0)
            if token in self.model:
                self.model[token] += context_sum
            # token not yet in model --> context vector equals context sum
            else:
                self.model[token] = context_sum

    def __store_model(self):
        """
        Stores the RI model, i.e. its model attribute (dict) as a .pkl
        file. The path is generated using the RI instance's string
        representation and is stored in the models directory.
        If a model already exists, a warning will be presented and the
        model will be overridden, as even though hyperparameters are
        identical, the model will differ due to random indexing.
        """
        dir_path = os.path.join("..", "models")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        filename = str(self) + ".pkl"
        path = os.path.join("..", "models", filename)
        if os.path.exists(path):
            print(
                Fore.RED + f"Warning: Model ('{self}') already exists with "
                "the given hyperparameters. \nIt will be overridden "
                "as versions will still differ due to random indexing."
                + Fore.RESET
            )
        with open(path, "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(Fore.RED + f"Model '{self}' was successfully created and"
                             f" stored in {path}" + Fore.RESET)


def load_model(path):
    """
    Loads an RI model from a given .pkl file.
    @param path: Valid path leading to a .pkl file.
    @return: RI model as dict.
    """
    assert os.path.exists(path), \
        "Invalid model path. Must be a path leading to a valid .pkl file" \
        " in the 'models' directory"
    with open(path, "rb") as f:
        rim_model = pickle.loads(f.read())
    return rim_model


if __name__ == '__main__':
    # example usage
    test_corpus = ["This", "is", "a", "small", "test", "corpus"]
    model = RandomIndexingModel(2, 2, 500, 100, test_corpus, name="test")
    loaded_model = load_model(os.path.join("..", "models",
                                           "model_test_2_2_500_100.pkl"))
