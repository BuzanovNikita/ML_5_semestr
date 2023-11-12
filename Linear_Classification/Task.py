import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.features = dict()

        for feature in X.columns:
            self.features[feature] = dict()
            for i, category in enumerate(np.sort(X[feature].unique())):
                self.features[feature][category] = i
        return self.features

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        result = []

        def one_hot_encoding(x):
            elem = np.array([], dtype=int)
            for feature, category in x.items():
                tmp = np.zeros(len(self.features[feature].keys()), dtype=int)
                tmp[self.features[feature][category]] = 1
                elem = np.concatenate((elem, tmp))
            result.append(elem)

        X.apply(one_hot_encoding, axis=1)

        return np.array(result)

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.features = dict()
        for feature in X.columns:
            self.features[feature] = dict()
            for category in np.sort(X[feature].unique()):
                successes = Y[X[feature] == category].mean()
                counters = X[X[feature] == category].shape[0] / X.shape[0]
                self.features[feature][category] = [successes, counters]

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        result = []

        def one_hot_encoding(x):
            elem = []
            for i, j in x.items():
                tmp = np.concatenate((self.features[i][j], [(self.features[i][j][0] + a) / (self.features[i][j][1] + b)]))
                elem = np.concatenate((elem, tmp))
            result.append(elem)

        X.apply(one_hot_encoding, axis=1)
        return np.array(result)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:
    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.folds = []
        self.array_of_dicts = []
        gener = group_k_fold(X.shape[0], self.n_folds, seed=seed)

        for idx_fold, idx_train in gener:
            self.folds.append(idx_fold)
            Train = X.iloc[idx_train, :]
            Target_train = Y.to_frame().iloc[idx_train, 0]

            features = dict()
            for feature in X.columns:
                features[feature] = dict()
                for category in np.sort(Train[feature].unique()):
                    successes = Target_train[Train[feature] == category].mean()
                    counters = Train[Train[feature] == category].shape[0] / Train.shape[0]
                    features[feature][category] = [successes, counters]
            self.array_of_dicts.append(features)

        self.folds = np.array(self.folds)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """

        def folds_simple_encoding(x):
            elem = []
            for i, j in x.items():
                tmp = np.concatenate((features[i][j], [(features[i][j][0] + a) / (features[i][j][1] + b)]))
                elem = np.concatenate((elem, tmp))
            res_tmp.append(elem)

        flag = True

        for i in range(self.n_folds):
            features = self.array_of_dicts[i]
            Fold = X.iloc[self.folds[i], :]
            res_tmp = []

            Fold.apply(folds_simple_encoding, axis=1)
            res_tmp = np.array(res_tmp)
            if flag:
                result = res_tmp
                flag = False
            else:
                result = np.concatenate((result, res_tmp))

        idx = self.folds.flatten()
        idx, result = zip(*[(b, a) for b, a in sorted(zip(idx, result))])

        return np.array(result)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """

    features = dict()
    for i, category in enumerate(np.sort(np.unique(x))):
        features[category] = i

    X = []

    for i in range(len(x)):
        elem = np.zeros(len(features.keys()), dtype=int)
        elem[features[x[i]]] = 1
        X.append(elem)
    X = np.array(X)

    w = []
    for i in range(X.shape[1]):
        tmp = X[y == 1]
        tmp = tmp[tmp[:, i] == 1]
        n = len(tmp)
        tmp = X[y == 0]
        tmp = tmp[tmp[:, i] == 1]
        m = len(tmp)
        w.append(n / (n + m))

    return np.array(w)
