import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class SMOGN:
    """Class performe over and under sampling for regression data.

    This Object is implementation of SMOGN
    """

    def __init__(self, threshold, over_sampling_ratio, under_sampling_ratio, k, dist=euclidean_distances):
        self.threshold = threshold
        self.over_sampling_ratio = over_sampling_ratio
        self.under_sampling_ratio = under_sampling_ratio
        self.k = k
        self.dist = dist

    def fit(self, X, y):
        self.relevances = self.relevance_fn(y)
        X_bins_n = X[y < self.relevances]
        y_bins_n = y[y < self.relevances]
        X_bins_r = X[y >= self.relevances]
        y_bins_r = y[y > self.relevances]
        X_newD = X_bins_r
        y_newD = y_bins_r

        if self.under_sampling_ratio > 0.:
            for Xn, yn in zip(X_bins_n, y_bins_n):

        if self.over_sampling_ratio > 0.:

    def relevance_fn(self, y, k=0.5, eps=1e-8):
        y_tilda = np.median(y)
        miny = np.min(y)
        maxy = np.max(y)
        ordD = np.argsort(y)

        adjL = np.vectorize(lambda x: max(x, 0))(ordD - 1)
        adjH = np.vectorize(lambda x: min(x, len(y) - 1))(ordD + 1)

        i = y[adjL] * (y[adjL] - miny) / (y - miny + eps)
        d = y[adjH] * (y[adjH] - y_tilda) / (maxy - y_tilda + eps)

        m = (np.abs(y_tilda - (y[adjL] + i)) +
             np.abs(y_tilda - (y[adjH] - d))) / 2.

        xa = y[adjL]
        xb = y[adjL] + k * m
        xc = y[adjH]
        xd = y[adjH] - k * m

        c1 = np.exp((xa * np.log(1. / 3.)) / (xa - xb))
        c2 = np.exp((xc * np.log(1. / 3.)) / (xc - xd))

        s1 = np.log(1. / 3.) / (xa - xb)
        s2 = np.log(1. / 3.) / (xc - xd)
        relevances = np.where(y <= y_tilda, c1 /
                              (c1+np.exp(s1*y)), c2/(c2+np.exp(s2*y)))
        return relevances
