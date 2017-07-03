# Tensor Factorization
This is a python implementation of tensor factorization (tucker decomposition) using SGD and ALS.

## Input Data Format
Each line presents one rating infomation and each column presents one rating feature.
The last column is rating value (float) and the rest are features (integer).
For example, the data below has two ratings (rating = 2 and 4) and three features. 
```
Feat1,Feat2,Feat3,Rating
    0,    3,    1,    2
    1,    2,    2,    4
```

## Quick Start
```
$ python TF.py --train *train_file* --test *test_file* --reg 0.1 --regS 0.1 --lr 0.1 --lr 0.01
```
You can type **python TF.py --help** for more details about the parameters.

## Reference
```
Karatzoglou, Alexandros, et al. Multiverse recommendation: n-dimensional tensor factorization for context-aware collaborative filtering. Proceedings of the fourth ACM conference on Recommender systems. ACM, 2010.
```
