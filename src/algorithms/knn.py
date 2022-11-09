from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import autosklearn.classification
import pickle
from src.algorithms.Detector import Detector


class KNN(Detector):
    def __init__(self, doload=False, n_neighbors=4, algorithm='brute'):
        super(KNN, self).__init__()
        self.doload = doload
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
        return

    def fit(self, x, y, count):
        if self.doload:
            self.knn = self.load('./src/temp/knn_%d.pickle' % count)
        else:
            self.knn.fit(x.copy(), y.copy())
            self.save(self.knn, './src/temp/knn_%d.pickle' % count)

    def predict(self, x):
        outlierscore = self.knn.predict_proba(x)[:, 1]
        return outlierscore


