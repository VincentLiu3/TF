# coding: utf-8

import sktensor
import numpy as np
from functools import reduce
from utils import read_data, multiply_list, integrate_type
from datetime import datetime

def CPTF(X, Xtest, dims, rank, reg=0.1, reg_S=0.1, t=1000000, tol=0, batch_ratio=0.5, max_epo=10):
	'''
	X, Xtest = 2D matrix
	dims, rank = used to init core and U
	rank = # of factors for each feature
	'''
	assert(X.ndim == 2), "CPTF: X must be a 2D matrix."
	assert(Xtest.ndim == 2), "CPTF: Xtest must be a 2D matrix."

	ninstance = X.shape[0]
	ndims = X.shape[1]-1
	max_iter = int(max_epo * ninstance)
	batch_size = int(ninstance * batch_ratio)

	iter_count = 0
	pre_loss = -1

	trn_y = X.T[ndims]
	tst_y = Xtest.T[ndims]

	# initialze core and U
	if isinstance(rank, int):
		rank = [rank for _ in range(ndims)] # num of facotr for each feature
	
	core, U = init_factors(rank, dims)
	
	start_time = datetime.now()
	tic = datetime.now()
	while iter_count <= max_iter:
		# pick on instance in X randomly
		for i in np.random.permutation(ninstance): 
			iter_count += 1

			# compute eps
			learn_rate = 1 / np.sqrt(t)
			t += 1

			ind = X[i][0:ndims]
			val = X[i][ndims]

			# compute F_ijk and f - y
			Ui_list = [U[k][ind[k]] for k in range(ndims)]
			F_ijk = tensor_ttv(core, Ui_list)
			e_ijk = F_ijk - val
			
			for k in range(ndims):
				step = tensor_ttv(core, Ui_list, k)
				U[k][ind[k]] = U[k][ind[k]] - learn_rate * reg * U[k][ind[k]] - learn_rate * e_ijk * step
				Ui_list = [U[k][ind[k]] for k in range(ndims)]


				'''
				change += learn_rate * e_ijk * np.linalg.norm(step.flat)
				if learn_rate * e_ijk * np.linalg.norm(step.flat) > 10:
					print('-------update U_i-------')
					print('learn_rate={}. e_ijk={}'.format(learn_rate, e_ijk))
					print(step)
					print(-1 * learn_rate * e_ijk * step)
				'''

			step = reduce(np.multiply.outer, Ui_list)
			core = core - learn_rate * reg * core - learn_rate * e_ijk * step
			
			
			'''
			change += learn_rate * e_ijk * np.linalg.norm(step.flat)
			if  > 10:
				print('-------update core-------')
				print('Ui_list =')
				print(Ui_list)
				print('learn_rate={}. e_ijk={}'.format(learn_rate, e_ijk))
				print('step = ')
				print(-1 * learn_rate * e_ijk * step)
			'''

			if iter_count%batch_size==0:
				print("[CPTF] {}/{} iterations done. ".format(iter_count, max_iter))

				# trn_pred_y = pred(X, core, U)
				# loss = OptLoss(trn_orin_y, trn_pred_y, core, U, reg, reg_S)
				tst_pred_y = pred(Xtest, core, U)
				loss = OptLoss(tst_y, tst_pred_y, core, U, reg, reg_S)
				print('[CPTF] Loss = {:.2f}. Change = {:.2f}'.format(loss, loss - pre_loss))

				# tst_pred_y = pred(Xtest, core, U)
				test_rmse = RMSE(tst_y, tst_pred_y)
				test_mae = MAE(tst_y, tst_pred_y)
				print("[CPTF] RMSE = {:.4f}. MAE = {:.4f}".format(test_rmse, test_mae))

				toc = datetime.now() - tic
				print("[CPTF] Running time = {:.1f}s".format(toc.total_seconds()))

				if loss - pre_loss > tol and pre_loss > 0:
					print("[CPTF] Early Stop!")
					iter_count = max_iter + 1
					break
				
				pre_loss = loss
				tic = datetime.now()

	total_time = datetime.now() - start_time
	print("[CPTF] CPTF fininshed. Total running time = {:.1f}s".format(total_time.total_seconds()))
	return core, U

def tensor_ttv(core, Ui_list, remove_k=None):
	'''
	Tensor times vector
	core = tensor
	Ui_list = list of U_i to be multiplied
	remove_k = do not multiply in k-th mode
	return a scalar or 1D array
	-----------
	use sktensor library
	'''
	dt_core = sktensor.dtensor(core)
	assert( dt_core.ndim == len(Ui_list) ), "tensor_ttv: core and U must be tha same dimension."

	modes = ()
	for k in range(dt_core.ndim):
		if k is not remove_k:
			modes += (k,)

	'''
	modes = range(dt_core.ndim)
	if remove_k is not None:
		modes.remove(remove_k)
	'''

	out_core = dt_core.ttv(tuple(Ui_list), modes)
	'''
	if np.linalg.norm(out_core.flat) > 200:
		print('-----')
		print(Ui_list)
		print(modes)
		print(core)
		print(out_core)
	'''
	if len(out_core) == 1: # out_core is a scaler
		return( out_core[0] )
	else:
		return( np.array(out_core) )

def OptLoss(X, Y, core, U, reg, reg_S):
	'''
	X = orign value (np array or list)
	Y = predicted value (np array or list)
	return square loss
	'''
	assert( len(X)==len(Y) ), "LossFunction: two different size arrays."
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
	for i in range(ninstance): # Complexity: O( n * k^k )
		ind = X[i][0:ndims]
		val = X[i][ndims]

		Ui_list = [U[k][ind[k]] for k in range(ndims)]
		pred_Y = tensor_ttv(core, Ui_list)
		prediction.append(pred_Y)
	
	return( prediction )

def init_factors(rank, dims):
	'''
	Initialize tensor and matrix with small values
	'''
	assert(len(rank) == len(dims)), "init_factors: Rank must be the same length as dimensions."
	Core = np.random.rand(multiply_list(rank)).reshape(rank) / multiply_list(rank)
	U = [np.random.rand(dims[i], rank[i])/rank[i] for i in range(len(dims))]
	return Core, U

if __name__ == "__main__":
	X, tst_dims = read_data("data/train3.txt")
	Xtest, trn_dims = read_data("data/test3.txt")
	dims = integrate_type(tst_dims, trn_dims)

	print("[CPTF] Numbe of training data = {}".format(X.shape[0]))
	print("[CPTF] Numbe of testing data = {}".format(Xtest.shape[0]))
	print("[CPTF] Number of types for each feature = {}".format(dims))

	rank = [10, 10, 10]
	reg = 0.001
	reg_S = 0.005
	t = 1000000
	print("------------Running CPTF------------")
	print("Rank = {}. reg for U = {}. reg for core = {}. initial learn rate = {}".format(rank, reg, reg_S, t))
	core, U = CPTF(X, Xtest, dims, rank, reg, reg_S, t)
	# test_err = RMSE(Xtest, core, U) 
	# print ("[CPTF] Final RMSE = {}".format(test_err))

# 140.112.187.116