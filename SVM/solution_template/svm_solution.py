import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_test_scaled = scaler.transform(test_features)
    best_model = SVC(kernel="rbf", C=1438449888.2876, gamma=0.00054555947, class_weight="balanced")

    best_model.fit(X_train_scaled, train_target)
    y_pred = best_model.predict(X_test_scaled)

    return np.array(y_pred)
