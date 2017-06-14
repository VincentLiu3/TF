import argparse
import sktensor
import time
import numpy as np
from functools import reduce
from utils import *

def parse_args():
	parser = argparse.ArgumentParser(description = 'Tensor Factorization')
	parser.add_argument('--train' , type = str, default = 'data/ml-1m/train.txt', help = 'Training file')
	parser.add_argument('--test' , type = str, default = 'data/ml-1m/test.txt', help = 'Testing file')
	parser.add_argument('--out' , type = str, default = '', help = 'File where the final result will be saved')

	parser.add_argument('--k', type = str, default = '10', help = 'Dimension of latent fectors, e.g. \'10-10-10\'')
	parser.add_argument('--reg', type = float, default = 0.1, help = 'Regularization for latent facotrs')
	parser.add_argument('--regS', type = float, default = 0.1, help = 'Regularization for core tensor')
	parser.add_argument('--lr', type = int, default = 1000000, help = 'Initial learning rate for latent facotrs')
	parser.add_argument('--lrS', type = int, default = 1000000, help = 'Initial learning rate for core tensor')
	
	parser.add_argument('--batchRatio', type = float, default = 0.1, help = 'Training instances for each iteration')
	parser.add_argument('--maxEpo', type = int, default = 2, help = 'Max training epo')
	parser.add_argument('--verbose', type = int, default = 1, help = 'Verbose or not')
	return parser.parse_args()

def CPTF(X, Xtest, dims, rank, reg, reg_S, t, t_S, batch_ratio, max_epo, verbose, tol=0):
	'''
	X, Xtest = n-dimension tensor presented in 2-dimension matrix 
	dims, rank = used to init core and U
	rank = # of factors for each feature
	'''
	assert(X.ndim == 2 and Xtest.ndim == 2), "CPTF: X must be a 2D matrix."
		
	ninstance = X.shape[0]
	ndims = X.shape[1]-1
	max_iter = int(max_epo * ninstance)
	batch_size = int(ninstance * batch_ratio)

	iter_count = 0
	pre_loss = -1

	trn_y = X.T[ndims]
	tst_y = Xtest.T[ndims]

	# initialze core and U
	core, U = init_factors(rank, dims)
	
	while iter_count < max_iter:
		# pick on instance in X randomly
		tic = time.time()
		trn_loss = 0
		for i in np.random.permutation(ninstance):
			iter_count += 1

			# Compute learn rate
			lr = 1 / np.sqrt(t)
			lr_S = 1 / np.sqrt(t_S)
			t += 1
			t_S += 1

			ind = X[i][0:ndims]
			val = X[i][ndims]

			# Compute F_ijk and f - y
			Ui_list = [U[k][ind[k]] for k in range(ndims)]
			F_ijk = tensor_ttv(core, Ui_list)
			e_ijk = F_ijk - val
			trn_loss += pow(e_ijk, 2)

			# Update U
			for k in range(ndims):
				step = lr * reg * U[k][ind[k]].copy()
				step += lr * e_ijk * tensor_ttv(core, Ui_list, k)
				U[k][ind[k]] -= step
				Ui_list = [U[k][ind[k]] for k in range(ndims)]

			# Update core tensor
			step = lr_S * reg_S * core
			step += lr_S * e_ijk * reduce(np.multiply.outer, Ui_list)
			core -= step
			
			# Evaluation
			if iter_count%batch_size==0:
				trn_loss = training_loss(trn_loss, core, U, reg, reg_S)
				change_rate = (pre_loss - trn_loss) / pre_loss * 100

				if verbose == 1:
					toc = time.time()
					print("[TF] Iter {}/{}. Time: {:.1f}".format(iter_count, max_iter, toc - tic))
					print('[TF] Training Loss = {:.2f} (change {:.2f}%)'.format(trn_loss, change_rate))

				'''
				## Testing Loss
				tst_pred_y = pred(Xtest, core, U)
				tst_loss = testing_loss(tst_y, tst_pred_y, core, U, reg, reg_S)
				test_rmse = RMSE(tst_y, tst_pred_y)
				
				if verbose == 1:
					toc = time.time()
					print('[TF] Testing Loss = {:.2f} (change {:.2f}%)'.format(tst_loss, change_rate))
					print("[TF] RMSE = {:.4f}".format(test_rmse))
					print("[TF] Time = {:.1f}s".format(toc - tic))
				'''

				if change_rate < tol and iter_count > 1:
					print("[TF] Early Stoping due to insufficient change in training loss!")
					iter_count = max_iter
					break
				
				pre_loss = trn_loss

	return core, U

def tensor_ttv(core, Ui_list, remove_k=None):
	'''
	Tensor times vector product (use sktensor library)
	-----------
	core = core tensor
	Ui_list = list of U_i to be multiplied
	remove_k = do not multiply in k-th mode
	return a scalar or np.array
	'''
	dt_core = sktensor.dtensor(core)
	assert( dt_core.ndim == len(Ui_list) ), "tensor_ttv: core and U must be tha same dimension."

	modes = ()
	for k in range(dt_core.ndim):
		if k is not remove_k:
			modes += (k,)

	out_core = dt_core.ttv(tuple(Ui_list), modes)

	if len(out_core) == 1: # out_core is a scaler
		return( out_core[0] )
	else:
		return( np.array(out_core) )

def training_loss(loss, core, U, reg, reg_S):
	U_l2 = [np.linalg.norm(Ui.flat) for Ui in U]
	core_l1 = np.linalg.norm(core.flat, ord = 1)
	core_l2 = np.linalg.norm(core.flat)
	return( loss / core_l1 + reg * sum(U_l2) + reg_S * core_l2 )

def testing_loss(X, Y, core, U, reg, reg_S):
	'''
	X: orign value (np array or list)
	Y: predicted value (np array or list)
	return: objective loss
	'''
	assert( len(X)==len(Y) ), "ObjLoss: two different size arrays."
	U_l2 = [np.linalg.norm(Ui.flat) for Ui in U]
	core_l1 = np.linalg.norm(core.flat, ord = 1)
	core_l2 = np.linalg.norm(core.flat)
	return( sum( pow(X-Y, 2) ) / core_l1 + reg * sum(U_l2) + reg_S * core_l2 )

def RMSE(X, Y):
	'''
	X, Y = np.array or list
	'''
	assert( len(X)==len(Y) ), "RMSE: Two arrays must be the same size."
	return( np.sqrt( sum( pow(X-Y, 2) ) / len(X) ) )

def MAE(X, Y):
	'''
	X, Y = np.array or list
	'''
	assert( len(X)==len(Y) ), "MAE: Two arrays must be the same size."
	return( sum( abs(X-Y) ) / len(X) )

def pred(X, core, U):
	'''
	X = orign value (np array or list)
	core = sparse tensor
	U = list of matrix
	return prediction  
	'''
	ninstance = X.shape[0]
	ndims = X.shape[1]-1

	prediction = []
	for i in range(ninstance): # Complexity: O( ninstance * k^k )
		ind = X[i][0:ndims]

		Ui_list = [U[k][ind[k]] for k in range(ndims)]
		pred_Y = tensor_ttv(core, Ui_list)
		prediction.append(pred_Y)
		
	return( prediction )

def init_factors(rank, dims):
	'''
	Initialize tensor and matrix with small random values
	'''
	assert(len(rank) == len(dims)), "Rank must be the same length as data dimensions."
	Core = np.random.rand(multiply_list(rank)).reshape(rank) / multiply_list(rank)
	U = [np.random.rand(dims[i], rank[i])/rank[i] for i in range(len(dims))]
	return Core, U

if __name__ == "__main__":
	args = parse_args()
	X, tst_dims = read_data(args.train)
	Xtest, trn_dims = read_data(args.test)
	dims = integrate_dims(tst_dims, trn_dims)
	rank = string2list(args.k, len(dims))

	if(args.verbose == 1):
		start_time = time.time()
		print('----------------- TF -----------------')
		print("[Data] Training Size = {}. Testing Size = {}	".format(X.shape[0], Xtest.shape[0]))
		print("[Data] Number of types for each feature = {}".format(dims))
		print("[Settings] K = {}. reg = {}. regS = {}. lr = {}. lrS =  {}".format(rank, args.reg, args.regS, args.lr, args.lrS))

	# Training
	core, U = CPTF(X, Xtest, dims, rank, args.reg, args.regS, args.lr, args.lrS, args.batchRatio, args.maxEpo, args.verbose)
	
	# Evaluation
	pred = pred(Xtest, core, U)
	orin = Xtest.T[Xtest.shape[1]-1]
	rmse = RMSE(orin, pred)
	save_result(args, rmse)

	if(args.verbose == 1):
		end_time = time.time()
		print('[Result] RMSE = {:.4f}'.format(rmse))
		print("[Result] Total Time = {:.1f}s".format(end_time - start_time))
	