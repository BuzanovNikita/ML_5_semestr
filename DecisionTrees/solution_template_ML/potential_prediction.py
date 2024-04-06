import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import numpy as np


def get_edge(x, axis=0, rot=False):
    if rot:
        x = np.rot90(x, 2)
    argmin = np.argmin(x, axis=axis)
    argmin = argmin[argmin > 0]
    if len(argmin) == 0:
        result = 0 if not rot else 256
    else:
        result = np.min(argmin) if not rot else 256 - np.min(argmin)
    return result


def center(x):
    top = get_edge(x)
    left = get_edge(x, axis=1)
    bottom = get_edge(x, rot=True)
    right = get_edge(x, axis=1, rot=True)

    x_center = (top + bottom) // 2
    y_center = (left + right) // 2

    x = np.roll(x, 128 - x_center, axis=0)
    x = np.roll(x, 128 - y_center, axis=1)
    if 128 - x_center < 0:
        x[128 - x_center:] = 20
    else:
        x[:128 - x_center] = 20

    if 128 - y_center < 0:
        x[128 - y_center:] = 20
    else:
        x[:128 - y_center] = 20
    return x


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        result = np.empty_like(x)
        for i in range(x.shape[0]):
            result[i] = center(x[i])
        return result.reshape((result.shape[0], -1))


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    rfr = RandomForestRegressor(random_state=42, n_estimators=1000, max_depth=9, max_features="sqrt", warm_start=True)
    transformer = PotentialTransformer()
    for i in range(4):
        X_train_transformed = transformer.fit_transform(X_train, Y_train)
        rfr.fit(X_train_transformed, Y_train)
        rfr.n_estimators += 1000
        for pot in range(X_train.shape[0]):
            X_train[pot] = np.rot90(X_train[pot], 1)
    for pot in range(X_train.shape[0]):
        X_train[pot] = np.flip(X_train[pot], axis=0)
    for i in range(4):
        X_train_transformed = transformer.fit_transform(X_train, Y_train)
        rfr.fit(X_train_transformed, Y_train)
        rfr.n_estimators += 1000
        for pot in range(X_train.shape[0]):
            X_train[pot] = np.rot90(X_train[pot], 1)
    del X_train_transformed
    del X_train
    X_test = transformer.transform(X_test)
    predictions = rfr.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
