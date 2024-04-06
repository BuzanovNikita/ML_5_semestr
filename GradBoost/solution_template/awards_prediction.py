import pandas as pd
from catboost import CatBoostRegressor
from numpy import ndarray
from sklearn.preprocessing import MultiLabelBinarizer


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    y_train = df_train["awards"]
    del df_train["awards"]

    df_train.fillna(-999, inplace=True)
    df_test.fillna(-999, inplace=True)

    df_train = df_train.replace('UNKNOWN', 'Some')
    df_test = df_test.replace('UNKNOWN', 'Some')

    for column in ['runtime', 'filming_locations', 'directors']:
        del df_train[column]
        del df_test[column]

    for i in range(3):
        del df_train[f"actor_{i}_gender"]
        del df_test[f"actor_{i}_gender"]

    mlb = MultiLabelBinarizer()
    genres_onehot = mlb.fit_transform(df_train['keywords'])
    genres_onehot_df = pd.DataFrame(genres_onehot, columns=mlb.classes_)
    df_train = pd.concat([df_train, genres_onehot_df], axis=1)
    df_train.drop(['keywords'], axis=1, inplace=True)

    test_genres_onehot = mlb.transform(df_test['keywords'])
    test_genres_onehot_df = pd.DataFrame(test_genres_onehot, columns=mlb.classes_)
    df_test = pd.concat([df_test, test_genres_onehot_df], axis=1)
    df_test.drop(['keywords'], axis=1, inplace=True)

    mlb = MultiLabelBinarizer()
    genres_onehot = mlb.fit_transform(df_train['genres'])
    genres_onehot_df = pd.DataFrame(genres_onehot, columns=mlb.classes_)
    df_train = pd.concat([df_train, genres_onehot_df], axis=1)
    df_train.drop(['genres'], axis=1, inplace=True)

    test_genres_onehot = mlb.transform(df_test['genres'])
    test_genres_onehot_df = pd.DataFrame(test_genres_onehot, columns=mlb.classes_)
    df_test = pd.concat([df_test, test_genres_onehot_df], axis=1)
    df_test.drop(['genres'], axis=1, inplace=True)

    cbr = CatBoostRegressor(n_estimators=1094, max_depth=7, learning_rate=0.0731, train_dir='/tmp/catboost_info',
                            logging_level='Silent')
    cbr.fit(df_train.to_numpy(), y_train.to_numpy())
    return cbr.predict(df_test.to_numpy())
