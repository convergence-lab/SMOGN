# SMOGN
# Author: Masashi Kimura

import pandas as pd
import numpy as np
from copy import copy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


class SMOGN:
    """Class performe over and under sampling for regression data.

    This Object is implementation of SMOGN

    Attribuites:
        threshold (float):
            threshold of rare example. [0, 1]

        over_sampling_ratio (float):
            ratio of over sample rare example. [0, 1]

        under_sampling_ratio (float):
            ratio of under sample normal example. [0, 1]

        k (int):
            number of nearest neighbors

        relevanse_base (float):
            base parameter of relevance_fn

        pert (float):
            pertubation parameter of gaussian noise

        metric (str):
            metric of distance.

        relevances (np.array):
            relevance values
    """

    def __init__(self, threshold=0.5, over_sampling_ratio=1.0, under_sampling_ratio=1.0, k=10, relevanse_base=0.5, pert=0.02, metric="minkowski"):
        """
        initialize SMOGN

        Args:
            threshold (float):
                threshold of rare example. [0, 1]

            over_sampling_ratio (float):
                ratio of over sample rare example. [0, 1]

            under_sampling_ratio (float):
                ratio of under sample normal example. [0, 1]

            k (int):
                number of nearest neighbors

            relevanse_base (float):
                base parameter of relevance_fn

            pert (float):
                pertubation parameter of gaussian noise

            metric (str):
                metric of distance.
        """
        self.threshold = threshold
        self.over_sampling_ratio = over_sampling_ratio
        self.under_sampling_ratio = under_sampling_ratio
        self.k = k
        self.relevance_base = relevanse_base
        self.metric = metric
        self.pert = pert

    def fit_transform(self, X, target_column):
        """
        Args:
            X (pd.DataFrame):
                training examples
        Returns:
            newX (pd.DataFrame):
                new training examples
        """
        self.relevances = self.relevance_fn(X[target_column].values.astype(np.float32), self.relevance_base)

        encoder = LabelEncoder()
        categorical_columns = []
        nominal_columns = []
        for col in X:
            if X[col].dtype == "object":
                categorical_columns += [col]
            else:
                nominal_columns += [col]
        for col in categorical_columns:
            X[col] = encoder.fit_transform(X[col])

        Xn = X.iloc[self.relevances < self.threshold]
        Xr = X.iloc[self.relevances > self.threshold]
        newD = [Xr]

        if self.under_sampling_ratio < 1.:
            sample_num = int(len(Xr) * self.under_sampling_ratio)
            normal_case = Xn.sample(n=sample_num)
            newD += [normal_case]
        else:
            newD += [Xn]

        if self.over_sampling_ratio > 0.:
            sample_num = int(len(Xn) * self.over_sampling_ratio)
            nn = NearestNeighbors(metric=self.metric)
            nn.fit(X)
            new_samples = []
            i = 0
            dist, neighbors_idx = nn.kneighbors(Xr, n_neighbors=self.k + 1)
            for i in range(sample_num):
                idx = np.random.randint(len(Xr))
                rare_sample = Xr.iloc[idx]
                rnd_i = np.random.randint(1, self.k + 1)
                dist_i, idx_i = dist[idx][rnd_i], neighbors_idx[idx][rnd_i]
                neighbor_sample = X.iloc[idx_i]
                maxD = np.median(dist[idx]) / 2.
                std = np.std(X)
                new_sample = pd.Series([0.] * len(X.columns), index=X.columns)
                for ci, col in enumerate(X):
                    if col in categorical_columns:
                        items = list(set(X[col]))
                        new_sample[col] = items[np.random.randint(len(items))]
                    else:
                        if dist_i < maxD:
                            diff = rare_sample[col] - neighbor_sample[col]
                            # shift = np.random.rand() * diff
                            shift = np.clip(np.abs(np.random.randn()), 0, 1) * diff
                            new_sample[col] = rare_sample[col] + shift
                        else:
                            pert = min(maxD, self.pert)
                            new_sample[col] = rare_sample[col] + np.random.randn() * std[ci] * pert
                new_samples += [new_sample]
            newD += [pd.DataFrame(new_samples)]
        newD = pd.concat(newD)
        return newD

    def relevance_fn(self, y, k=0.5, eps=1e-8):
        """
        calcuate relevance

        Args:
            y: np.array
                target examples

            k: float
                relevance base

            esp: float
                small value to avoid zero divition
        Returns:
        relevance values: np.array
            relevance values of y
        """
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
