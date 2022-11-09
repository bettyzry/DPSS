import evaluation
import utils
from src.algorithms.Detector import Detector
import numpy as np


class HoltWinter(Detector):
    def __init__(self, weight1=1, weight2=1, weight3=1, alpha=0.5, beta=0.5, types='linear', gama=0.5, m=1):
        super(HoltWinter, self).__init__()
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3

        self.alpha = alpha
        self.beta = beta
        self.types = types
        self.gama = gama
        self.m = m
        self.history = np.empty((0, ))

    def predict_proba(self, x):
        ts = np.array(x)
        size, feature = ts.shape
        outlierscores = np.zeros([size, feature])
        for ii in range(feature):
            d = ts[:, ii]
            score = self.single_predict(d)
            outlierscores[:, ii] = score
        # print(self.weight1, self.weight2, self.weight3)
        if self.weight3 == 0 and self.weight2 == 0 and self.weight1 == 0:
            outlierscore = np.zeros(len(outlierscores))
        else:
            outlierscore = np.average(outlierscores, axis=1, weights=(self.weight1, self.weight2, self.weight3))
        outlierscore = np.array([[1-i, i] for i in outlierscore])
        return outlierscore

    def single_predict(self, data):
        Y = data
        types = self.types

        if types == 'linear':
            alpha, beta = self.alpha, self.beta
            a = [Y[0]]
            b = [Y[1] - Y[0]]
            y = [a[0] + b[0]]
            for i in range(len(Y)):
                a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                y.append(a[i + 1] + b[i + 1])
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gama
            m = self.m
            a = [sum(Y[0:m]) / float(m)]
            b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]

            if types == 'additive':
                s = [Y[i] - a[0] for i in range(m)]
                y = [a[0] + b[0] + s[0]]
                for i in range(len(Y)):
                    a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
                    b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                    s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
                    y.append(a[i + 1] + b[i + 1] + s[i + 1])

            elif types == 'multiplicative':
                s = [Y[i] / a[0] if Y[i] / a[0] != 0 else 0.0001 for i in range(m)]
                y = [(a[0] + b[0]) * s[0]]
                for i in range(len(Y)):
                    a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
                    b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                    s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
                    y.append((a[i + 1] + b[i + 1]) * s[i + 1])
            else:
                raise ValueError("ERROR: unsupported type, expect linear, additive or multiplicative.")
        y.pop()

        diff_value = abs(data - y)
        diff_value_normalize = utils.normalize(abs(diff_value))

        if sum(np.isnan(diff_value_normalize)):
            print('!!!')
        return diff_value_normalize

