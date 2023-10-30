import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    result = []

    size_of_fold = num_objects // num_folds
    full_indexes = np.arange(0, num_objects)

    for i in range(0, num_folds-1):
        valid_indexes = np.arange(i * size_of_fold, (i + 1) * size_of_fold)
        train_indexes = np.setdiff1d(full_indexes, valid_indexes)
        result.append((train_indexes, valid_indexes))

    valid_indexes = np.arange((num_folds - 1) * size_of_fold, num_objects)
    train_indexes = np.setdiff1d(full_indexes, valid_indexes)
    result.append((train_indexes, valid_indexes))
    return result


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    result = dict()
    for normalizer in parameters['normalizers']:
        for neighbors in parameters['n_neighbors']:
            for metrics in parameters['metrics']:
                for weights in parameters['weights']:
                    data = X.copy()
                    target = y.copy()
                    model = knn_class(n_neighbors=neighbors, weights=weights, metric=metrics)
                    score = 0
                    for iter in range(len(folds)):
                        if normalizer[1] != "None":
                            normalizer[0].fit(X[folds[iter][0]])
                            data = normalizer[0].transform(X)
                        model.fit(data[folds[iter][0]], target[folds[iter][0]])
                        target_pred = model.predict(data[folds[iter][1]])
                        score += score_function(target[folds[iter][1]], target_pred)
                    result[(normalizer[1], neighbors, metrics, weights)] = score / len(folds)
    return result
