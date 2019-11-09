# SMOGN
# Author: Masashi Kimura

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


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

    def __init__(self, threshold=0.95, over_sampling_ratio=0.1, under_sampling_ratio=1.0, k=10, relevanse_base=0.5, pert=0.02, metric="minkowski"):
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

    def fit_transform(self, X, y):
        """
        Args:
            X (np.array): 
                training examples

            y (np.array): 
                target examples

        Returns:
            newX (np.array): 
                new training examples

            newy (np.array): 
                new target examples
        """
        self.relevances = self.relevance_fn(y, self.relevance_base)
        B = np.hstack([X, np.expand_dims(y, 1)])
        Bn = B[self.relevances < self.threshold]
        Br = B[self.relevances > self.threshold]
        newD = [Br]

        if self.under_sampling_ratio < 1.:
            sample_num = int(len(Br) * self.under_sampling_ratio)
            inds = np.arange(len(Bn), dtype=np.int)
            normal_case = Bn[np.random.choice(inds, size=sample_num)]
            newD += [normal_case]
        else:
            newD += [Bn]

        if self.over_sampling_ratio > 0.:
            sample_num = int(len(Bn) * self.over_sampling_ratio)
            nn = NearestNeighbors(metric=self.metric)
            nn.fit(B)
            new_samples = []
            i = 0
            dist, neighbors_idx = nn.kneighbors(Br, n_neighbors=self.k + 1)
            for i in range(sample_num):
                idx = np.random.randint(len(Br))
                rare_sample = Br[idx]
                rnd_i = np.random.randint(1, self.k + 1)
                dist_i, idx_i = dist[idx][rnd_i], neighbors_idx[idx][rnd_i]
                neighbor_sample = B[idx_i]
                maxD = np.median(dist[idx]) / 2.
                std = np.std(B)
                if dist_i < maxD:
                    diff = rare_sample - neighbor_sample
                    #shift = np.random.rand() * diff
                    shift = np.clip(np.abs(np.random.randn()), 0, 1) * diff
                    new_sample = rare_sample + shift
                    new_samples += [new_sample]
                else:
                    pert = min(maxD, self.pert)
                    new_sample = rare_sample + np.random.randn() * std * pert
                    new_samples += [new_sample]
            newD += [new_samples]
        newD = np.vstack(newD)
        newX = newD[:, :-1]
        newy = newD[:, -1]
        return newX, newy

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
