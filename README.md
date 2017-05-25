# Tensor Factorization

This is an implementation of n-dimensional tensor factorization using SGD and ALS to minimize square loss. 

## Input Format
Each line presents one rating data and each column presents one rating feature.
The last column is rating value and the rest are features. 
For example, the data below has two data (rating = 2 and 4), and each ratings have three features. 
```
X1,X2,X3,Y
0,3,1,2
1,2,2,4
```

## Reference
* Karatzoglou, Alexandros, et al. Multiverse recommendation: n-dimensional tensor factorization for context-aware collaborative filtering. Proceedings of the fourth ACM conference on Recommender systems. ACM, 2010.
