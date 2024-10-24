#!/usr/bin/env python

"""
Builds class vectors based on document vectors, runs the classification and
computes precision and recall.

This script expects as input csv files that contains one document vector per
line and a separate file containing the corresponding labels:
-t: The training set of document representations, one vector per line
-l: The labels of the training set documents, one per line
-d: The documents to be classified
-g: The true labels of the documents, for evaluation


mini example
docs = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
genres = np.array(["humor", "humor", "religion"])

vectors = build_class_vectors(docs, genres)

result = evaluate_class_vectors(
    [[3, 4, 5, 6],[10, 20, 30, 40],[0, 1, 2, 3],[0, 5, 2, 4],[1, 2, 3, 4],[1, 2, 3, 4],[9, 10, 11, 12]],
    ["humor", "humor", "religion", "humor", "religion", "humor", "religion"],
    vectors
)

for k, v in result.items():
    print(k, v, sep='\n')

"""

import click
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def build_class_vectors(documents, classes):
    """Nimmt eine Liste von Dokumentvektoren und dazugehörigen Klassen und baut
    für jede Klasse einen Klassenvektor.

    :param documents: array or list, Dokumentenvektoren
    :param classes: array or list, Klassenbezeichnungen für die Dokumente in
           documents.

    :return: dict, ein Klassenvektor je Klasse mit den Klassennamen als keys
    """

    assert len(documents) == len(classes)

    return {
        label: np.average(
            [documents[i] for i in range(len(documents))
             if classes[i] == label],
            axis=0)
        for label in set(classes)
    }


def evaluate_class_vectors(documents, gold, class_vectors):
    """Nimmt eine Liste von Dokumentvektoren und eine Liste von Klassen, aus
    denen die ähnlichste bestimmt werden soll.

    :param documents: array or list, Dokumentenvektoren
    :param gold: array or list, die tatsächlichen Klassen der Dokumente
    :param class_vectors: dict, Mapping von Klassen auf Klassenvektoren
    :return: dict, eine Übersicht der Ergebnisse
    """
    assert len(documents) == len(gold)

    predicted = []
    for document in documents:
        closest_class = None
        best_measure = None
        for label, vector in class_vectors.items():
            similarity = cosine_similarity(document, vector)
            if closest_class is None or similarity > best_measure:
                best_measure = similarity
                closest_class = label
        predicted.append(closest_class)

    # order of elements in view objects is guaranteed to be insertion order
    # starting at Python3.7 but let's be sure just in case someone uses
    # an older version
    labels = list(set(class_vectors.keys()))

    return compile_results(gold, predicted, labels)


def compile_results(gold, predicted, labels):

    return {
        'labels': labels,
        'gold': gold,
        'predicted': predicted,
        'confusion_matrix': confusion_matrix(gold, predicted, labels=labels),
        'scores_by_label': precision_recall_fscore_support(
            gold, predicted, zero_division=0, average=None, labels=labels
            ),
        'scores_overall': precision_recall_fscore_support(
            gold, predicted, zero_division=0, average='macro'
            )
    }


def print_results(result):

    print('=== Confusion Matrix ===\n')
    print(result['labels'])
    print(result['confusion_matrix'])

    # render confusion matrix that normalizes over the true labels
    disp = ConfusionMatrixDisplay.from_predictions(
        result['gold'],
        result['predicted'],
        xticks_rotation=90,
        normalize='true'
        )
    plt.show()

    print('\n===OVERALL RESULTS===\n')
    print(f'P={round(result["scores_overall"][0],2)}, '
          f'R={round(result["scores_overall"][1],2)}, '
          f'F={round(result["scores_overall"][2],2)}')
    print()

    print('\n===RESULTS PER LABEL===\n')
    for i in range(len(result['labels'])):
        print(f'Class={result["labels"][i]:{30}} '
              f'P={round(result["scores_by_label"][0][i],2):.2f}, '
              f'R={round(result["scores_by_label"][1][i],2):.2f}, '
              f'F={round(result["scores_by_label"][2][i],2):.2f}')


def cosine_similarity(vector_a, vector_b):
    """Berechnet die Kosinusähnlichkeit zwischen zwei Vektoren.

    :return: float, für ternäre RI-Vektoren eine Zahl zwischen -1 und 1,
             wobei 1, wenn vector_a==vector_b
    """
    return 1 - distance.cosine(vector_a, vector_b)


def read_from_file(filename):
    try:
        return np.loadtxt(filename, delimiter=',')
    except ValueError:
        return np.loadtxt(filename, delimiter=',', dtype=str)


@click.command()
@click.option('-t', 'train',
              type=click.Path(exists=True),
              required=True,
              help='The training set of document representations.')
@click.option('-l',
              'labels',
              type=click.Path(exists=True),
              required=True,
              help='The labels of the training set documents.')
@click.option('-d',
              'documents',
              type=click.Path(exists=True),
              required=True,
              help='The documents to be classified.')
@click.option('-g',
              'gold',
              type=click.Path(exists=True),
              required=True,
              help='The true labels of the documents.')
def classify_and_evaluate(train, labels, documents, gold):
    train_docs = read_from_file(train)
    train_classes = read_from_file(labels)
    docs = read_from_file(documents)
    classes = read_from_file(gold)

    model = build_class_vectors(train_docs, train_classes)
    result = evaluate_class_vectors(docs, classes, model)
    print_results(result)


if __name__ == '__main__':
    classify_and_evaluate()
