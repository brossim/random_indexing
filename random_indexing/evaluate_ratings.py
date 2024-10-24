# Author        : Simon Bross
# Date          : March 8, 2023
# Python version: 3.9
# File purpose  : Terminal interface for intrinsic evaluation

import click
from IntrinsicEvaluator import IntrinsicEvaluator
from RandomIndexingModel import load_model
from colorama import Fore


@click.command()
@click.option('-m', '--model', 'model',
              type=click.Path(exists=True),
              help='Valid path leading to a model stored as a .pkl file',
              required=True)
@click.option('-r', '--ratings', 'ratings_path',
              type=click.Path(exists=True),
              required=True,
              help='Valid path leading to either simlex or wordsim '
                   'dataset stored in a .txt file')
def main(model, ratings_path):
    """
    Provides a terminal interface for the RI models' intrinsic evaluation
    using either simlex or wordsim datasets to compute the spearman correlation
    between the human ratings and the cosine similarity of word pairs. Cf.
    class IntrinsicEvaluator for implementation and further documentation.
    @param model: Valid path leading to a model stored as a .pkl file.
    @param ratings_path: Valid path leading to either simlex or wordsim dataset
    stored in a .txt file.
    """
    ri_model = load_model(model)
    evaluator = IntrinsicEvaluator(ratings_path, ri_model)
    print(Fore.RED + f"Evaluation results for '{model}' using {ratings_path}: "
          f"{Fore.GREEN}{evaluator.compute_spearman()}" + Fore.RESET)


if __name__ == '__main__':
    main()
