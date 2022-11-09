import numpy as np
import pandas as pd

import evaluation


def normalize(array):
    _ave_ = np.average(array)
    _var_ = np.var(array)

    if _var_ != 0:
        norm_array = np.array([(arr - _ave_) / _var_ for arr in array])
    else:
        norm_array = np.array([(arr - _ave_) for arr in array])
    return norm_array


def min_max_to_one(array):
    _min_ = min(array)
    _max_ = max(array)
    norm_array = np.array([(arr-_min_) / (_max_ - _min_) for arr in array])
    return norm_array


def score2label_threshold(score, percentage=0.95):
    import math
    score = np.array(score)
    temp_array = score.copy()
    temp_array.sort()
    if percentage == 0:
        threshold = temp_array[0]
    else:
        threshold = temp_array[math.floor(len(temp_array) * percentage)-1]

    label = np.zeros(len(score), dtype=int)
    for ii, s in enumerate(score):
        if s > threshold:
            label[ii] = 1
    return label


def score2label_thresholdvalue(score, threshold=0.95):
    score = np.array(score)
    temp_array = score.copy()
    temp_array.sort()

    label = np.zeros(len(score), dtype=int)
    for ii, s in enumerate(score):
        if s >= threshold:
            label[ii] = 1
    return label


def data_standardize2(df):
    mini, maxi = df.min(), df.max()
    for col in df.columns:
        if maxi[col] != mini[col]:
            df[col] = (df[col] - mini[col]) / (maxi[col] - mini[col])


def data_standardize(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):
    mini, maxi = X_train.min(), X_train.max()
    for col in X_train.columns:
        if maxi[col] != mini[col]:
            X_train[col] = (X_train[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = (X_test[col] - mini[col]) / (maxi[col] - mini[col])
            # @TODO: the max and min value after min-max normalization is 1 and 0, so the clip doesn't work?
            X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
        else:
            assert X_train[col].nunique() == 1
            if remove:
                if verbose:
                    print("Column {} has the same min and max value in train. Will remove this column".format(col))
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
            else:
                if verbose:
                    print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                if mini[col] != 0:
                    X_train[col] = X_train[col] / mini[col]  # Redundant operation, just for consistency
                    X_test[col] = X_test[col] / mini[col]
                if verbose:
                    print("After transformation, train unique vals: {}, test unique vals: {}".format(
                    X_train[col].unique(),
                    X_test[col].unique()))
    return X_train, X_test


def test():
    ytrue = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
    score = [0.1, 0.2, 0.3, 0.4, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9]
    ypred = evaluation.get_best_label(ytrue, score)
    print(ypred)
    f1, p, r = evaluation.get_fpr(ytrue, ypred)
    print(f1, p, r)
    auc, pr, f1, p, r = evaluation.evaluation(score, ytrue)
    print(auc, pr, f1, p, r)


if __name__ == '__main__':
    test()