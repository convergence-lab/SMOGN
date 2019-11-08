import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


class SMOGN:
    """Class performe over and under sampling for regression data.

    This Object is implementation of SMOGN
    """

    def __init__(self, threshold, over_sampling_ratio, under_sampling_ratio, k, metric="minkowski"):
        self.threshold = threshold
        self.over_sampling_ratio = over_sampling_ratio
        self.under_sampling_ratio = under_sampling_ratio
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.relevances = self.relevance_fn(y)
        B = np.hstack(X, y)
        Bn = B[self.relevances < self.threshold]
        Br = B[self.relevances > self.threshold]
        newD = Br

        if self.under_sampling_ratio > 0.:
            sample_num = len(X) * self.under_sampling_ratio
            normal_case = np.random.choice(Bn, size=sample_num)
            newD += [normal_case]

        if self.over_sampling_ratio > 0.:
            sample_num = len(X) * self.over_sampling_ratio
            nn = NearestNeighbors(metric=self.metric)
            nn.fit(B)
            new_samples = []
            for bi in Br:
                print(i)
                dist, neighbors_idx = nn.kneighbors(bi, n_neighbors=self.k)
                maxD = np.median(dist) / 2
                for i in sample_num:
                    neighbors = np.hstack(dist, neighbors_idx)
                    sample = np.random.choice(neighbors)
                    dist, idx = sample
                    if dist < maxD:
                        new_samples += [bi]  # FIXME: use SMOTER
                    else:
                        pert = min(maxD, 0.02)
                        new_samples += [bi]  # FIXME : user Gaussian noise
            newD += np.array(new_samples)
        return np.hstack(newD)

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
                              (c1 + np.exp(s1 * y)), c2 / (c2 + np.exp(s2 * y)))
        return relevances
