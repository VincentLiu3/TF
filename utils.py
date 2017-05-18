import pandas as pd

def read_data(file_name):
	X = pd.read_csv(file_name, sep=',', header=None)
	X = X.as_matrix()
	ndims = X.shape[1]-1
	dims = [X.T[i].max()+1 for i in range(ndims)]
	return X, dims

def multiply_list(lst):
	product = 1
	for itm in lst:
		product *= itm
	return(product)

def integrate_type(dim1, dim2):
	assert( len(dim1) == len(dim2) ), "integrate_type: two dimensions must be the same length."
	dims = [max(dim1[i], dim2[i]) for i in range(len(dim1))]
	return(dims)