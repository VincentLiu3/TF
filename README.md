# Tensor Factorization
This is an implementation of n-dimensional tensor factorization using SGD and ALS to minimize loss function. 

## Input Data Format
Each line presents one rating infomation and each column presents one rating feature.
The last column is rating value and the rest are features.
For example, the data below has two ratings (rating = 2 and 4) three features. 
```
Feat1,Feat2,Feat3,Rating
0    ,3    ,1    ,2
1    ,2    ,2    ,4
```

## Reference
Karatzoglou, Alexandros, et al. Multiverse recommendation: n-dimensional tensor factorization for context-aware collaborative filtering. Proceedings of the fourth ACM conference on Recommender systems. ACM, 2010.
