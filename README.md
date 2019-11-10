# SMOGN

Python Implementation of SMOGN algorithm.

SMOGN is a over and under sampling algorighm for regression data.

```
git clone --recursive https://github.com/convergence-lab/SMOGN.git
```

[Docmentation](https://convergence-lab.github.io/SMOGN/index.html#document-index)


## Examples

```
python test.py                                                                                                                (MobileInteractNet) 4171ms  土 11/ 9 12:58:58 2019
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/Abalone.csv
RandomForrest           MSE: 4.23
SMOGN RandomForrest     MSE: 4.0
Relative Improvement    5.66 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/Accel.csv
RandomForrest           MSE: 14.7
SMOGN RandomForrest     MSE: 17.8
Relative Improvement    -19.2 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/a1.csv
RandomForrest           MSE: 3.69e+02
SMOGN RandomForrest     MSE: 3.6e+02
Relative Improvement    2.28 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/a2.csv
RandomForrest           MSE: 62.8
SMOGN RandomForrest     MSE: 65.9
Relative Improvement    -4.92 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/a3.csv
RandomForrest           MSE: 47.5
SMOGN RandomForrest     MSE: 46.5
Relative Improvement    1.97 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/a4.csv
RandomForrest           MSE: 5.78
SMOGN RandomForrest     MSE: 3.02
Relative Improvement    62.9 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/a6.csv
RandomForrest           MSE: 1.91e+02
SMOGN RandomForrest     MSE: 1.57e+02
Relative Improvement    19.3 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/a7.csv
RandomForrest           MSE: 10.9
SMOGN RandomForrest     MSE: 7.81
Relative Improvement    32.9 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/availPwr.csv
RandomForrest           MSE: 5.24e+03
SMOGN RandomForrest     MSE: 3.71e+03
Relative Improvement    34.2 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/bank8FM.csv
RandomForrest           MSE: 0.00113
SMOGN RandomForrest     MSE: 0.00116
Relative Improvement    -1.98 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/boston.csv
RandomForrest           MSE: 23.0
SMOGN RandomForrest     MSE: 17.9
Relative Improvement    24.9 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/cpuSm.csv
RandomForrest           MSE: 10.5
SMOGN RandomForrest     MSE: 10.1
Relative Improvement    3.39 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/fuelCons.csv
RandomForrest           MSE: 6.14
SMOGN RandomForrest     MSE: 6.51
Relative Improvement    -5.86 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/heat.csv
RandomForrest           MSE: 2.31e+02
SMOGN RandomForrest     MSE: 2.33e+02
Relative Improvement    -0.789 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/maxTorque.csv
RandomForrest           MSE: 1.48e+04
SMOGN RandomForrest     MSE: 1.26e+04
Relative Improvement    15.7 %

Avg Improvemnet 11.4 %
```


## References

- Branco, P., Ribeiro, R. P., Torgo, L., Krawczyk, B., & Moniz, N. (2017). SMOGN: a Pre-processing Approach for Imbalanced Regression. In Proceedings of Machine Learning Research (Vol. 74).
