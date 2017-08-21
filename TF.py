import argparse
import sktensor
import logging
import time
import numpy as np
from functools import reduce
from utils import *

def parse_args():
	parser = argparse.ArgumentParser(description = 'Tensor Factorization')
	parser.add_argument('--train' , type = str, default = '', help = 'Training file')
	parser.add_argument('--test' , type = str, default = '', help = 'Testing file')

	parser.add_argument('--model' , type = str, default = '', help = 'Directory where the final model will be saved')
	parser.add_argument('--out' , type = str, default = '', help = 'File where the final result will be saved')

	parser.add_argument('--k', type = str, default = '8', help = 'Dimension of latent fectors, e.g. \'8-8-8\'')
	parser.add_argument('--reg', type = float, default = 0.1, help = 'Regularization for latent facotrs')
	parser.add_argument('--regS', type = float, default = 0.1, help = 'Regularization for core tensor')
	parser.add_argument('--lr', type = float, default = 0.1, help = 'Initial learning rate for latent facotrs')
	parser.add_argument('--lrS', type = float, default = 0.1, help = 'Initial learning rate for core tensor')
	parser.add_argument('--maxEpo', type = int, default = 10, help = 'Max training epo')
	
	parser.add_argument('--verbose', type = int, default = 1, help = 'Verbose or not')
	return parser.parse_args()

def TPTF(X, Xtest, dims, rank, reg, reg_S, lr, lrS, max_iter, tol=0):
	'''
	Tucker decomposition with ALS and SGD
	-----------------------------------
	X, Xtest = n-dimension tensor presented in [2D matrix, 1D array]
	dims, rank = used to init core and U
	rank = # of factors for each feature
	'''
	assert(len(X) == 2 and len(Xtest) == 2), "TPTF: no rating values."
	assert(X[0].ndim == 2 and Xtest[0].ndim == 2), "TPTF: X must be a 2D matrix."
	assert(X[0].shape[0] == len(X[1]) and Xtest[0].shape[0] == len(Xtest[1])), "TPTF: rating and index must have the same length."
	
	t = int( pow( 1/lr, 2) )
	t_S = int( pow( 1/lrS, 2) )
 
	trn_x = X[0] # index of training data
	tst_x = Xtest[0] # index of testing data

	trn_y = X[1] # rating values of training data
	tst_y = Xtest[1] # rating values of testing data

	ninstance = trn_x.shape[0]
	ndims = trn_x.shape[1]
	
	iter_count = 0
	pre_loss = -1

	# initialze core and U
	core, U = init_factors(rank, dims)
	
	tic = time.time()
	while iter_count < max_iter:
		iter_count += 1
		trn_loss = 0
		# Training 
		for i in np.random.permutation(ninstance): # pick on instance in X randomly
			# Compute learn rate
			lr = 1 / np.sqrt(t)
			lrS = 1 / np.sqrt(t_S)
			t += 1
			t_S += 1

			ind = trn_x[i]
			val = trn_y[i]

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
			step_S = lrS * reg_S * core.copy()
			step_S += lrS * e_ijk * reduce(np.multiply.outer, Ui_list)
			core -= step_S

		# Training Loss
		trn_loss = training_loss(trn_loss, core, U, reg, reg_S)
		change_rate = (trn_loss - pre_loss) / pre_loss * 100
		
		## Testing Loss
		tst_pred_y = pred(Xtest[0], core, U)
		test_rmse = RMSE(tst_y, tst_pred_y)
		# tst_loss = testing_loss(tst_y, tst_pred_y, core, U, reg, reg_S)
		
		toc = time.time()
		logger.info('Iter {}/{}. Time: {:.1f}'.format(iter_count, max_iter, toc - tic))
		logger.info('Training Loss: {:.2f} (change {:.2f}%). Testing RMSE: {:.4f}'.format(trn_loss, change_rate, test_rmse))
		tic = time.time()

		''' Early Stopping
		if change_rate < tol and pre_loss > 0:
			print("[TF] Early Stoping due to insufficient change in training loss!")
			iter_count = max_iter
			break
		'''
		
		if np.isnan(trn_loss): # numpy overflow, need to use less learning rate
			logger.info('TF: Overflow')
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
	#
	modes = range(len(Ui_list))
	if remove_k is not None:
		modes.remove(remove_k)
	#
	out_core = dt_core.ttv(tuple(Ui_list), modes)
	#
	if len(out_core) == 1: # out_core is a scaler
		return( out_core[0] )
	else:
		return( np.array(out_core) )

def training_loss(loss, core, U, reg, reg_S):
	U_l2 = [np.linalg.norm(Ui.flat) for Ui in U]
	core_l1 = np.linalg.norm(core.flat, ord = 1)
	core_l2 = np.linalg.norm(core.flat)
	return( loss / core_l1 + reg / 2 * sum(U_l2) + reg_S / 2 * core_l2 )

def testing_loss(X, Y, core, U, reg, reg_S):
	'''
	X: orign value (np array or list)
	Y: predicted value (np array or list)
	return: objective loss
	'''
	assert( len(X)==len(Y) ), "Loss: Two different size arrays."
	U_l2 = [np.linalg.norm(Ui.flat) for Ui in U]
	core_l1 = np.linalg.norm(core.flat, ord = 1)
	core_l2 = np.linalg.norm(core.flat)
	return( sum( pow(X-Y, 2) ) / core_l1 + reg / 2 * sum(U_l2) + reg_S / 2 * core_l2 )
 
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

def pred(ind_X, core, U):
	'''
	X = index of test data (sparse matrix)
	core = core tensor (dense tensor)
	U = list of latent factors for each mode
	--------------------
	return prediction (list)
	'''
	ninstance = ind_X.shape[0]
	ndims = ind_X.shape[1]
	#
	prediction = []
	for i in range(ninstance): # Complexity: O( ninstance * k^k )
		ind = ind_X[i]
		Ui_list = [U[k][ind[k]] for k in range(ndims)]
		pred_Y = tensor_ttv(core, Ui_list)
		prediction.append(pred_Y)
	#
	return( prediction )

def init_factors(rank, dims):
	'''
	Initialize tensor and matrix with small random values
	'''
	assert(len(rank) == len(dims)), "Rank must be the same length as number of features."
	Core = np.random.rand(multiply_list(rank)).reshape(rank) # / multiply_list(rank)
	U = [np.random.rand(dims[i], rank[i])/rank[i] for i in range(len(dims))]
	return Core, U

if __name__ == "__main__":
	args = parse_args()
	X, trn_dims = read_data(args.train)
	Xtest, tst_dims = read_data(args.test)
	dims = integrate_dims(tst_dims, trn_dims)
	rank = string2list(args.k, len(dims))

	if(args.verbose == 1):
		logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
	else:
		logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
	
	start_time = time.time()
	logger = logging.getLogger()
	logger.info('----------------- TF -----------------')
	logger.info('Data: Number of types for each feature = {}'.format(dims))
	logger.info('Data: Training Size = {}. Testing Size = {}'.format(len(X[1]), len(Xtest[1])))
	logger.info('Settings: K = {}. reg = {}. regS = {}. lr = {}. lrS = {}'	.format(rank, args.reg, args.regS, args.lr, args.lrS))

	# Training
	core, U = TPTF(X, Xtest, dims, rank, args.reg, args.regS, args.lr, args.lrS, args.maxEpo)
	save_model(args, core, U)

	# Evaluation
	pred = pred(Xtest[0], core, U)
	rmse = RMSE(Xtest[1], pred)
	save_result(args, rmse)

	end_time = time.time()
	logger.info('RMSE: {:.4f}'.format(rmse))
	logger.info('Total Time: {:.1f}s'.format(end_time - start_time))
