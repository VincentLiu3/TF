# coding: utf-8

## Multiverse Recommendation Tensor Factorization
## Vincent Liu
import sktensor
import numpy as np
from functools import reduce
from utils import read_data, multiply_list, integrate_type
from datetime import datetime

def CPTF(X, Xtest, dims, rank, reg=0.001, reg_S=0.01, t=5000000, tol=-1, batch_size=50000, max_epo=10):
	'''
	X, Xtest = 2D matrix
	dims, rank = used to init core and U
	rank = # of factors for each feature
	'''
	assert(X.ndim == 2), "CPTF: X must be a 2D matrix."
	assert(Xtest.ndim == 2), "CPTF: Xtest must be a 2D matrix."

	ninstance = X.shape[0]
	ndims = X.shape[1]-1
	max_iter = max_epo * ninstance
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

			ind = X[i][0:ndims]
			val = X[i][ndims]

			# compute eps
			learn_rate = 1 / np.sqrt(t)
			t += 1

			# compute F_ijk and f - y
			Ui_list = [U[k][ind[k]] for k in range(ndims)]
			F_ijk = tensor_ttv(core, Ui_list)
			e_ijk = F_ijk - val
			
			for k in range(ndims):
				step = tensor_ttv(core, Ui_list, k)
				old_Ui = U[k][ind[k]]
				U[k][ind[k]] = old_Ui - learn_rate * reg * old_Ui - learn_rate * e_ijk * step
				Ui_list = [U[k][ind[k]] for k in range(ndims)]
				'''
				if learn_rate * e_ijk * np.linalg.norm(step.flat) > 10:
					print('-------update U_i-------')
					print('learn_rate={}. e_ijk={}'.format(learn_rate, e_ijk))
					print(step)
					print(-1 * learn_rate * e_ijk * step)
				'''

			step = reduce(np.multiply.outer, Ui_list)
			old_core = core
			core = old_core - learn_rate * reg * old_core - learn_rate * e_ijk * step
			'''
			if learn_rate * e_ijk * np.linalg.norm(step.flat) > 10:
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
				test_error = RMSE(tst_y, tst_pred_y)
				print("[CPTF] RMSE = {:.4f}".format(test_error))

				toc = datetime.now() - tic
				print("[CPTF] Running time = {:.2f}s".format(toc.total_seconds()))

				#if pre_loss - loss < tol and pre_loss > 0:
				#	total_time = datetime.now() - start_time
				#	print('[CPTF] Early stop. Total running time = {:.2f}s'.format(total_time.total_seconds()))
				#	return core, U
				
				pre_loss = loss
				tic = datetime.now()

	total_time = datetime.now() - start_time
	print("[CPTF] CPTF fininshed. Total running time = {:.2f}s".format(total_time.total_seconds()))
	return core, U

def tensor_ttv(core, Ui_list, remove_k=None):
	'''
	Tensor times vector
	core = tensor
	Ui_list = list of U_i to be multiplied
	remove_k = do not multiply in k-th mode
	-----------
	use sktensor library
	'''
	dt_core = sktensor.dtensor(core)
	assert( dt_core.ndim == len(Ui_list)), "tensor_ttv: core and U must be tha same dimension."

	modes = ()
	for k in range(dt_core.ndim):
		if k is not remove_k:
			modes += (k,)

	Ui_list = tuple(Ui_list)
	out_core = dt_core.ttv(Ui_list, modes, without=False)
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

	'''
	for k in range(len(Ui_list)):
		if i == remove_k: # do nothing in this iteration
			continue

		U_i = Ui_list[i]
		print('---------------------')
		print('U_i = {}'.format(U_i))
		print('out_core = {}'.format(out_core))
		print('out_core.ndim = {}'.format(out_core.ndim))
		print('k = {}'.format(k))

		if out_core.ndim > 1:
			out_core = out_core.ttv(U_i, k)
		elif out_core.ndim == 1:
			out_core = np.dot(out_core, U_i)
	'''

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
	assert( len(X)==len(Y) ), "RMSE: two different size arrays."
	return( np.sqrt( sum( pow(X-Y, 2) ) / len(X) ) )

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
	X, tst_dims = read_data("data/train.txt")
	Xtest, trn_dims = read_data("data/test.txt")
	dims = integrate_type(tst_dims, trn_dims)

	print("[CPTF] Numbe of training data = {}".format(X.shape[0]))
	print("[CPTF] Numbe of testing data = {}".format(Xtest.shape[0]))
	print("[CPTF] Number of types for each feature = {}".format(dims))

	rank = [10, 10, 3, 3, 3]
	reg=0.001
	reg_S=0.01
	t=5000000
	print("------Running CPTF------")
	print("[CPTF] Rank = {}. reg for U = {}. reg for core = {}. initail learn rate = {}".format(rank, reg, reg_S, t))
	core, U = CPTF(X, Xtest, dims, rank, reg, reg_S, t)
	# test_err = RMSE(Xtest, core, U) 
	# print ("[CPTF] Final RMSE = {}".format(test_err))

	
# 140.112.187.116