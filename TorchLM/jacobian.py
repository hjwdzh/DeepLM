import torch
import numpy as np
from .listvec import *
import subprocess
from time import time

def get_gpu_memory_map():
	"""Get the current gpu usage.

	Returns
	-------
	usage: dict
	    Keys are device ids as integers.
	    Values are memory usage as integers in MB.
	"""
	result = subprocess.check_output(
		[
			'nvidia-smi', '--query-gpu=memory.used',
			'--format=csv,nounits,noheader'
		], encoding='utf-8')
	# Convert lines into a dictionary
	gpu_memory = [int(x) for x in result.strip().split('\n')]
	gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
	return gpu_memory_map

def JacobiSquaredColumnNorm(jacobi, indices, variables):
	columnNorm = ListZero(variables)
	for i in range(len(variables)):
		index = indices[:,i]
		jx = jacobi[i]
		jx2 = torch.sum(jx * jx, dim = 0)
		columnNorm[i].index_add_(0, index, jx2)
		torch.cuda.empty_cache()

	return columnNorm

def JacobiNormalize(jacobi, indices, variables):
	jacobiScale = JacobiSquaredColumnNorm(jacobi, indices, variables)

	for i in range(len(jacobiScale)):
		jacobiScale[i] = 1.0 / (1.0 + torch.sqrt(jacobiScale[i]))
		column = jacobiScale[i][indices[:,i]]
		shapeDim = [i for i in column.shape]
		shapeDim.insert(0, jacobi[i].shape[0])
		column = column.unsqueeze(0).expand(shapeDim)
		jacobi[i] = jacobi[i] * column

	return jacobiScale

def LogJacobians(jacobians, x, y, indices):
	Jx = torch.zeros((jacobians[0].shape[0] * jacobians[0].shape[1],
		x.shape[0] * x.shape[1])).double().numpy()
	Jy = torch.zeros((jacobians[1].shape[0] * jacobians[1].shape[1],
		y.shape[0] * y.shape[1])).double().numpy()
	for resIdx in range(jacobians[0].shape[0]):
		for i in range(indices.shape[0]):
			Jx[i * jacobians[0].shape[0] + resIdx,
				(indices[i,0] * x.shape[1]):((indices[i,0] + 1) * x.shape[1])]\
				= jacobians[0][resIdx][i]
			Jy[i * jacobians[1].shape[0] + resIdx,
				(indices[i,1] * x.shape[1]):((indices[i,1] + 1) * x.shape[1])]\
				= jacobians[1][resIdx][i]

	J = np.concatenate([Jx, Jy], axis=1)
	print(J)

def JacobiBlockJtJ(jacobians, lmDiagonal, variables, indices, res):
	for r in res:
		r.zero_()
	for varid in range(len(jacobians)):
		jacobian = jacobians[varid]
		variableDim = jacobian.shape[2:]

		# residual_dim * loss_terms * variableDim
		jPlain = jacobian.view(jacobian.shape[0], jacobian.shape[1], -1)
		jtjS = torch.matmul(jPlain.permute(1, 2, 0), jPlain.permute(1, 0, 2))
		#jtj = torch.zeros(variables[varid].shape[0], jtjS.shape[1],
		#	jtjS.shape[2], dtype=torch.float64, device=variables[varid].device)
		res[varid].index_add_(0, indices[:,varid], jtjS)
		#jtjs.append(jtj)

	# how to vectorize?
	for varid in range(len(res)):
		diagonal = lmDiagonal[varid]
		diagonal = diagonal.view(diagonal.shape[0], -1)
		for vardim in range(diagonal.shape[1]):
			res[varid][:,vardim,vardim] += diagonal[:,vardim] ** 2

	#return jtjs

def JacobiLeftMultiply(jacobians, residuals, variables, indices, res):
	for r in res:
		r.zero_()

	for varid in range(len(variables)):
		jacobian = jacobians[varid]
		vshape = [i for i in jacobian.shape[2:]]
		j = jacobian.view(jacobian.shape[0], jacobian.shape[1], -1)
		j = j.permute(1,0,2)
		r = residuals.view(residuals.shape[0], residuals.shape[1], 1)
		r = r.expand(j.shape)
		jr = torch.sum(j * r, dim = 1)
		#torch.cuda.empty_cache()
		vshape.insert(0, jr.shape[0])
		res[varid].index_add_(0, indices[:,varid], jr.view(vshape))
		#del jr
	#torch.cuda.empty_cache()

	#return jtrs

def JacobiRightMultiply(jacobians, residuals, variables, indices, res):
	#jrs = None
	res.zero_()
	for varid in range(len(variables)):
		jacobian = jacobians[varid]
		residual = residuals[varid][indices[:,varid]]

		targetShape = [i for i in jacobian.shape]
		currentShape = [i for i in jacobian.shape]
		currentShape[0] = 1
		residual = residual.view(currentShape).expand(targetShape)

		res += torch.sum((residual * jacobian).view(jacobian.shape[0],
			jacobian.shape[1], -1),dim=2).permute(1, 0)

def JacobiJtJD(jacobians, diagonal, p, variables, indices, z, res):
	#torch.cuda.synchronize()
	#t1 = time()
	JacobiRightMultiply(jacobians, p, variables, indices, z)
	#torch.cuda.synchronize()
	#t2 = time()
	JacobiLeftMultiply(jacobians, z, variables, indices, res)
	#torch.cuda.synchronize()
	#t3 = time()
	for i in range(len(res)):
		res[i] += p[i] * (diagonal[i] ** 2)
	#torch.cuda.synchronize()
	#t4 = time()
	#print('jacobijtjd ', t2 - t1, t3 - t2, t4 - t3)
	#return [t[i] + p[i] * diagonal[i] ** 2 for i in range(len(t))]

def LogBlockJtJ(jtjs):
	dim = 0
	for j in jtjs:
		dim += j.shape[0] * j.shape[1]
	JtJ = np.zeros((dim, dim))
	dim = 0
	for j in jtjs:
		for k in range(j.shape[0]):
			JtJ[dim:dim+j.shape[1], dim:dim+j.shape[2]] = j[k]
			dim += j.shape[1]
	print(JtJ)

def PrepareSortedIndices(indices, variables):
	IndicesIdx = []
	EndIdx = []
	for i in range(len(variables)):
		I = indices[:,i]
		s = torch.argsort(I)
		ids, count = torch.unique(I[s], return_counts=True)
		count = torch.cumsum(count, dim = 0)

		IndicesIdx.append(s)
		EndIdx.append(count)

	return IndicesIdx, EndIdx

