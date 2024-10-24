# Author        : Simon Bross
# Date          : March 8, 2023
# Python version: 3.9
# File purpose  : Terminal interface for random indexing model training

import click
from RandomIndexingModel import RandomIndexingModel
from CorpusPreprocessor import CorpusPreprocessor


@click.command()
@click.option("--corpus", 'corpus', required=True,
              help="'brown' to access the entire brown corpus or a path "
                   "leading to a .txt file from which the corpus will"
                   " be created")
@click.option("-d", "--dimension", 'dimension', required=True,
              help="Vector dimension as int")
@click.option("-n", "--non_zero-dims", 'non_zero_dims', required=True,
              help='The number of vector dimension to be not zero, i.e. either'
                   ' 1 or -1')
@click.option("-l", "--left", 'left', required=True,
              help="The left context window size for the RI model")
@click.option("-r", "--right", "right", required=True,
              help="The right context window size for the RI model")
@click.option("--name", "name", required=False,
              help="Optional name influencing the RandomIndexingModel's string"
                   " representation and filename under which it is stored")
def main(corpus, dimension, non_zero_dims, left, right, name):
    """
    Provides a terminal interface for the creation of a random indexing model.
    Errors are thrown from the respective classes, unless the integer
    conversion of the main parameters fail.
    @param corpus: Corpus to be used for creating the RIM. Should be either
    'brown' to access the entire brown corpus from the nltk package or a path
    leading to a .txt file from which a corpus is generated.
    @param dimension: Number of dimensions for the index vectors in the RIM.
    @param non_zero_dims: Number of dimensions to be non-zero (i.e 1 or -1) in
    the RIM's index vectors, as int.
    @param left: Left window context size for the RIM, as int.
    @param right: Right window context size for the RIM, as int.
    @param name: String used to determine the RandomIndexingModel's string
    representation and filepath under which the model is stored, optional.
    If not provided, defaults to None.
    """
    preprocessor = CorpusPreprocessor(corpus=corpus)
    preprocessed_corp = preprocessor.preprocess_corpus()
    # model is stored automatically
    RandomIndexingModel(int(left), int(right), int(dimension),
                        int(non_zero_dims), preprocessed_corp, name=name)


if __name__ == '__main__':
    main()
