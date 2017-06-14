import numpy as np
import os.path

def read_data(file_name):
	X = np.loadtxt(file_name, dtype = int, delimiter = ',')
	ndims = X.shape[1]-1
	dims = [X.T[i].max()+1 for i in range(ndims)]
	return X, dims

def multiply_list(lst):
	product = 1
	for itm in lst:
		product *= itm
	return(product)

def integrate_dims(dim1, dim2):
	'''
	To avoid training and testing data with different dimensions.
	'''
	assert( len(dim1) == len(dim2) ), "Training and testing data must have the same number of features."
	dims = [max(dim1[i], dim2[i]) for i in range(len(dim1))]
	return(dims)
  
def save_result(args, rmse):
	'''
	save rmse to args.out
	'''
	if args.out != '':
		if os.path.exists(args.out) == True:
			with open(args.out, 'a') as fp_w:
				fp_w.write('{},{},{},{},{},{},{:.4f}\n'.format(args.k, args.reg, args.regS, args.lr, args.batchRatio, args.maxEpo, rmse))
		else:
			with open(args.out, 'w') as fp_w:
				fp_w.write('k,reg,regS,lr,lrS,batchRatio,maxEpo,RMSE\n')
				fp_w.write('{},{},{},{},{},{},{},{:.4f}\n'.format(args.k, args.reg, args.regS, args.lr, args.lrS, args.batchRatio, args.maxEpo, rmse))

def string2list(k, ndims):
	k_list = k.split('-')
	last_k = int(k_list[len(k_list)-1])
	rank = [int(k_list[x]) if (x < len(k_list)) else (last_k) for x in range(ndims)]
	return rank