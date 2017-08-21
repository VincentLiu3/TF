# Tensor Factorization
This is a python implementation of tensor factorization (tucker decomposition) using SGD and ALS.

## Input Data Format
Each row presents one rating infomation. The last column is rating value (float) and the rest are features (integer).
```
Feat1,Feat2,Feat3,Rating
    0,    3,    1,    2
    1,    2,    2,    4
```

## Quick Start
```
$ python TF.py --train data/ml-1m/user_train.txt --test data/ml-1m/user_test.txt --reg 0.1 --regS 0.1 --lr 0.1 --lr 0.1
```
You can type **python TF.py --help** for more details about the parameters.

## Reference
```
* Karatzoglou, Alexandros, et al. Multiverse recommendation: n-dimensional tensor factorization for context-aware collaborative filtering. Proceedings of the fourth ACM conference on Recommender systems. ACM, 2010.
```
