# Author        : Simon Bross
# Date          : March 9, 2023
# Python version: 3.9
# File purpose  : Implements a terminal interface for the creation of
#                 feature vectors (X) [document vectors] and its labels (y)
#                 for the extrinsic evaluation (exemplified by document
#                 classification) of random indexing models on the brown corpus


import click
import pathlib
from BrownDocVec import BrownDocVec
from RandomIndexingModel import load_model


@click.command()
@click.option('-p', '--path', 'path', type=click.Path(exists=True),
              help="Path leading to a RI model stored as a .pkl file in"
                   " the 'models' directory",
              required=True)
@click.option('-m', '--mode', 'mode',
              help="Mode to determine which data to be created. "
                   "Must be either 'training', 'validation', or 'testing'")
def main(path, mode):
    data_creator = BrownDocVec()
    model = load_model(path)
    # remove prefix 'models/' (or rather folder reference if stored
    # elsewhere) and suffix .pkl from model name irrespective of
    # the operating system
    model_name = path.removesuffix(".pkl")
    # split path into components and remove directory reference
    model_name = pathlib.Path(model_name).parts[-1]
    data_creator.create_csv_data(model, model_name, mode)


if __name__ == '__main__':
    main()
