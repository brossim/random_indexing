# Author        : Simon Bross
# Date          : February 27, 2023
# Python version: 3.9
# File purpose  : Implements a corpus preprocessor used before passing
#                 a corpus to the random indexing model


import re
import os
import string
import multiprocessing as mp
import time
from nltk.corpus.reader.util import ConcatenatedCorpusView
from nltk.corpus.reader.tagged import TaggedCorpusView
from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.data import find
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import download


class InvalidCorpusType(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


class CorpusPreprocessor:
    """
    Preprocesses a given corpus using the following techniques:
    0. Sentence and word tokenization: if .txt is provided.
    1. POS tagging: if a corpus is provided that is not already POS tagged.
    2. Stop word removal and removal of words tagged with 'X' (= 'other' in
       universal tagset; assumed to not provide relevant semantic information,
       e.g. words with spelling mistakes).
    3. Lemmatization: To group together inflection forms (e.g. sings --> sing)
    4. Removal of possessive 's' (as in president's)
    5. Lowercasing: To remove duplicates due to upper/lowercasing
    6. Removal of non-alphabetic tokens
    """

    def __init__(self, corpus=None):
        """
        Instantiates a CorpusPreprocessor instance.
        @param corpus: Corpus to be preprocessed, might be either 'brown',
        a supported iterable (list, ContatenatedCorpusView, TaggedCorpusView
        from nltk), or a path to a .txt file to be tokenized and preprocessed.
        If a supported iterable, it is distinguished between iterable(list)
        and iterable(list(tuple)), the latter being assumed to have already
        been POS tagged, the former to contain plain sentences that will be POS
        tagged.
        """
        # assure corpus parameter validity
        valid_types = (list, ConcatenatedCorpusView, str, TaggedCorpusView)
        if corpus != "brown" and not isinstance(corpus, valid_types):
            raise InvalidCorpusType("Invalid corpus type. "
                                    "Consult the class documentation.")

        # determines if corpus should be POS tagged
        # depending on the corpus parameter; default = True
        self.__pos_tagging = True

        # Procure all needed resources if not already installed
        if corpus == "brown":
            self.__prepare_components(check_brown=True)
        else:
            self.__prepare_components()

        # Case 1: already tokenized corpus, a supported iterable containing
        # lists (sentences)
        if isinstance(
                corpus, (list, ConcatenatedCorpusView, TaggedCorpusView)
        ):
            for sent in corpus:
                assert isinstance(sent, list), \
                    "Corpus must be composed of sublists representing the" \
                    " individual sentences"
                for ele in sent:
                    assert isinstance(ele, str) or isinstance(ele, tuple), \
                        "Every element in the corpus sublists must be a" \
                        " string or tuple containing the token and its" \
                        " respective POS tag"
                    # if ele is tuple, it is assumed that the corpus is already
                    # POS tagged (e.g. when passing another corpus from nltk
                    # on which .tagged_sents() was used
                    if isinstance(ele, tuple):
                        assert len(ele) == 2, \
                            "Tuples in the sublist must contain two elements" \
                            " (strings)"
                        for component in ele:
                            assert isinstance(component, str), \
                                "Tuples in the sublists must contain two " \
                                "strings, the token and its POS tag"
                        self.__pos_tagging = False
            self.__corpus = corpus

        # Case 2: entire brown corpus, e.g. for intrinsic evaluation
        elif corpus == "brown":
            self.__pos_tagging = False
            # use already tagged version to save computation time
            self.__corpus = brown.tagged_sents(tagset='universal')

        # Case 3: generic; custom corpus from valid .txt file
        elif isinstance(corpus, str):
            assert corpus.endswith(".txt") and os.path.exists(corpus), \
                "no .txt file or corpus .txt file could not be found"
            with open(corpus, "r", encoding="utf-8") as f:
                corpus_str = f.read()
                # determine sentence boundaries
                sent_tokenized = sent_tokenize(corpus_str)
                # tokenize every sentence
                self.__corpus = [
                    word_tokenize(sent) for sent in sent_tokenized
                ]

    @staticmethod
    def __prepare_components(check_brown=False):
        """
        Prepare and download - if needed - the necessary components
        for corpus preprocessing, i.e. the brown corpus, the
        universal tagset, the nltk stopwords and WordNetLemmatizer.
        @param check_brown: Boolean determining if brown corpus
        availability should be checked.
        """
        # Procure the brown corpus
        if check_brown:
            try:
                find(os.path.join("corpora", "brown"))
            except LookupError:
                download('brown')
        # Procure the necessary tagset (Universal Tagset) for POS Tagging
        try:
            find(os.path.join("taggers", "universal_tagset"))
        except LookupError:
            download("universal_tagset")
        # Procure the NLTK stopword list
        try:
            find(os.path.join("corpora", "stopwords"))
        except LookupError:
            download("stopwords")
        # Procure the WordNetLemmatizer data
        try:
            find(os.path.join("corpora", "omw-1.4"))
        except LookupError:
            download('omw-1.4', quiet=True)

    @property
    def corpus(self):
        """
        Returns the corpus.
        @return: Depending on the stage and input, list, ConcatenatedCorpusView
        or TaggedCorpusView.
        """
        return self.__corpus

    @corpus.setter
    def corpus(self, value):
        """
        Sets the corpus attribute.
        @param value: New corpus, depending on the stage and input,
        list, ConcatenatedCorpusView or TaggedCorpusView.
        """
        self.__corpus = value

    @staticmethod
    def _pos_tag(sent):
        """
        POS tags a sentence, using the universal tagset.
        @param sent: Sentence as list.
        @return: POS tagged sentence, i.e. a list of tuples(str,str).
        """
        return pos_tag(sent, tagset='universal')

    @staticmethod
    def isfloat(token):
        """
        Helper function to determine if a string token is a float.
        @param token: String.
        @return: Boolean.
        """
        try:
            float(token)
            return True
        except ValueError:
            return False

    def preprocess_corpus(self):
        """
        Preprocesses the corpus as explained in the class docstring.
        @return: Preprocessed corpus, as list.
        """
        # get stopwords and punctuation marks to be removed
        punctuation = string.punctuation.replace("", " ").split()
        nltk_stopwords = stopwords.words('english')
        all_stops = set(nltk_stopwords + punctuation)

        # map POS tags from universal tagset to its equivalents used by
        # WordNetLemmatizer and prepare lemmatizer
        universal_to_wnl = {
            "NOUN": "n",
            "VERB": "v",
            "ADJ": "a",
            "ADV": "r"
        }
        wnl = WordNetLemmatizer()

        # Step 1: POS-tagging before removing/lemmatizing any of the tokens
        # no POS tagging needed if brown corpus is used
        if self.__pos_tagging:
            with mp.Pool() as pool:
                self.corpus = pool.map(self._pos_tag, self.corpus)

        new_corpus = []
        for sent in self.corpus:
            # Step 2: remove stopwords, punctuation, digits, and words tagged
            # with 'X'; = other, i.e. "for words that for some reason cannot
            # be assigned a real part-of-speech category"
            new_sent = [
                (token, pos) for token, pos in sent
                if token not in all_stops and not token.isnumeric()
                and pos != "X" and not self.isfloat(token)
            ]
            for token, pos in new_sent:
                # Step 3: lemmatize tokens in order to group together related
                # words with the same meaning, i.e. reducing inflection forms
                # to their underlying lexeme, thus removing redundant morpho-
                # logical information (e.g. [she is] running / [he] runs
                # --> 'run') and reducing the semantic model's complexity
                try:
                    lemmatized_token = wnl.lemmatize(
                        token,
                        pos=universal_to_wnl[pos]
                    )
                # do not consider POS tag information if there is no mapping
                except KeyError:
                    lemmatized_token = wnl.lemmatize(token)

                # Step 4: remove "'s" from the end of token (e.g. president's)
                # to retrieve the desired token/lexeme
                pattern = re.compile(r"'s\b")
                new_token = re.sub(pattern, "", lemmatized_token)
                # Step 5: lowercase all tokens
                lowercased = new_token.lower()
                # Step 5: expand every token string by its POS tag,
                # e.g. can --> can_NOUN/can_VERB (Universal Tagset) in order to
                # distinguish between homographs that can be told apart using
                # POS tags. Only allow tokens that are alphabetic, thus also
                # removing unusual punctuation marks/invalid tokens etc. that
                # did not appear in all_stops
                if lowercased.isalpha():
                    new_corpus.append(lowercased + f"_{pos}")

        self.corpus = new_corpus
        return self.corpus


if __name__ == '__main__':
    # example usage
    start = time.time()
    test = CorpusPreprocessor(corpus="brown")
    test.preprocess_corpus()
    print(f"{time.time() - start:.2f}s")
    # duration: 6.33s
    num_tokens = len(test.corpus)
    num_types = len(set(test.corpus))
    print("Number of tokens: ", num_tokens)
    print("Number of types : ", num_types)
    # ratio might be lower than expected due to the POS-tag expanded tokens
    print("Type-Token-Ratio: ", f"{(num_types / num_tokens):.3f}")
