import numpy as np
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
import evaluation
import utils


class Detector(BaseEstimator, ClassifierMixin):
    def __init__(self):
        return

    def fit(self, x, y):
        return

    def predict(self, x, y=None):
        outlierscore = self.predict_proba(x)[:, 1]
        if y is None:
            pred_y = utils.score2label_threshold(outlierscore, 0.9)
        else:
            pred_y = evaluation.get_best_label(y, evaluation.adjust_scores(y, outlierscore))
        return pred_y

    def predict_proba(self, x):
        outlierscore = np.zeros(len(x))
        return outlierscore

    def save(self, detector, path):
        with open(path, 'wb') as f:
            pickle.dump(detector, f)  # 将训练好的模型存储在变量f中，且保存到本地

    def load(self, path):
        with open(path, 'rb') as f:
            detector = pickle.load(f)  # 将模型存储在变量clf_load中
            return detector