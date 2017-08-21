import numpy as np
import os

def read_data(file_name):
	'''
	X: index 
	Y: rating values
	'''
	X = np.loadtxt(file_name, dtype=float, delimiter=',')
	ndims = X.shape[1]-1
	Y = X.T[ndims] # rating values
	X = np.delete(X, ndims, 1).astype(int) # index values
	dims = [X.T[i].max()+1 for i in range(ndims)]
	return [X, Y], dims

def save_model(args, core, U):
	if args.model != '':
		if os.path.exists(args.model) is False:
			os.mkdir(args.model)  # e.g. models/ml-1m
		our_path = '{}/{}_{}_{}_{}_{}_{}_'.format(args.model, args.k, args.reg, args.regS, args.lr, args.lrS, args.maxEpo)  
		np.save(our_path+'core.npy', core)
		np.save(our_path+'U.npy', U)

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
		if os.path.exists(args.out) is False:
			with open(args.out, 'w') as fp: # create the file and write header
				fp.write('k,reg,regS,lr,lrS,maxEpo,RMSE\n')

		with open(args.out, 'a') as fp:
			fp.write('{},{},{},{},{},{},{:.4f}\n'.format(args.k, args.reg, args.regS, args.lr, args.lrS, args.maxEpo, rmse))
		
def string2list(k, ndims):
	k_list = k.split('-')
	last_k = int(k_list[len(k_list)-1])
	rank = [int(k_list[x]) if (x < len(k_list)) else (last_k) for x in range(ndims)]
	return rank