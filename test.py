import numpy as np
import pandas as pd


df = pd.read_csv("Imbalanced-Regression-DataSets/CSV_data/a1.csv")

X = df.drop("a1", axis=1).values
y = df["a1"].values
y_tilda = np.median(y)
miny = np.min(y)
maxy = np.max(y)
eps = 1e-8
k = 0.5

ordD = np.argsort(y)

adjL = np.vectorize(lambda x: max(x, 0))(ordD - 1)
adjH = np.vectorize(lambda x: min(x, len(y) - 1))(ordD + 1)

i = y[adjL] * (y[adjL] - miny) / (y - miny + eps)
d = y[adjH] * (y[adjH] - y_tilda) / (maxy - y_tilda + eps)

m = (np.abs(y_tilda - (y[adjL] + i)) + np.abs(y_tilda - (y[adjH] - d))) / 2.

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

for yi, r in zip(y, relevances):
    print(yi, "\t", r)
