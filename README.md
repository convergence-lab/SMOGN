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
RandomForrest           MSE: 4.53
SMOGN RandomForrest     MSE: 4.66
Relative Improvement    -2.77 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/Accel.csv
RandomForrest           MSE: 0.933
SMOGN RandomForrest     MSE: 0.685
Relative Improvement    30.6 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/fuelCons.csv
RandomForrest           MSE: 0.146
SMOGN RandomForrest     MSE: 0.153
Relative Improvement    -4.6 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/heat.csv
RandomForrest           MSE: 2.52
SMOGN RandomForrest     MSE: 2.64
Relative Improvement    -4.64 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/availPwr.csv
RandomForrest           MSE: 16.9
SMOGN RandomForrest     MSE: 14.8
Relative Improvement    12.9 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/boston.csv
RandomForrest           MSE: 30.8
SMOGN RandomForrest     MSE: 24.9
Relative Improvement    21.2 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/bank8FM.csv
RandomForrest           MSE: 0.00101
SMOGN RandomForrest     MSE: 0.000989
Relative Improvement    2.19 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/maxTorque.csv
RandomForrest           MSE: 34.7
SMOGN RandomForrest     MSE: 36.9
Relative Improvement    -6.17 %
=====================
Dataset: Imbalanced-Regression-DataSets/CSV_data/cpuSm.csv
RandomForrest           MSE: 9.2
SMOGN RandomForrest     MSE: 8.28
Relative Improvement    10.5 %

Avg Improvemnet 6.59 %
```


## References

- Wang, B., Zheng, H., Liang, X., Chen, Y., Lin, L., & Yang, M. (n.d.). Toward Characteristic-Preserving Image-based Virtual Try-On Network. Retrieved from https://github.com/sergeywong/cp-vton.
