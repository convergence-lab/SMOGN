import numpy as np
import pandas as pd
from smogn import SMOGN

df = pd.read_csv("Imbalanced-Regression-DataSets/CSV_data/a1.csv")

X = df.drop("a1", axis=1).values
y = df["a1"].values

sm = SMOGN(0.7, 1.0, 0.0, 5)
new = sm.fit(X, y)
print(new)
