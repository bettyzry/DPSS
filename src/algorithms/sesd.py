import evaluation
import utils
from src.algorithms.Detector import Detector
import numpy as np


class Sesd(Detector):
    def __init__(self, seasonality=206, weight1=1, weight2=1, weight3=1):
        super(Sesd, self).__init__()
        self.seasonality = seasonality
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3

    def predict_proba(self, x):
        import statsmodels.api as sm
        import scipy.stats as stats

        ts = np.array(x)
        size, feature = ts.shape
        seasonal = self.seasonality         # Seasonality is 20% of the ts if not given.

        outlierscores = np.zeros([size, feature])
        for ii in range(feature):
            d = ts[:, ii]
            decomp = sm.tsa.seasonal_decompose(d, period=seasonal)
            residual = d - decomp.seasonal - np.median(d)
            diff_value_normalize = abs(stats.zscore(residual, ddof=1))
            outlierscores[:, ii] = diff_value_normalize
        # print(self.weight1, self.weight2, self.weight3)
        if self.weight3 == 0 and self.weight2 == 0 and self.weight1 == 0:
            outlierscore = np.zeros(len(outlierscores))
        else:
            outlierscore = np.average(outlierscores, axis=1, weights=(self.weight1, self.weight2, self.weight3))
        outlierscore = np.array([[1-i, i] for i in outlierscore])
        return outlierscore