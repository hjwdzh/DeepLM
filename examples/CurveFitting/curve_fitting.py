import numpy as np
import torch
from TorchLM.solver import Solve

import matplotlib.pyplot as plt

# Ground truth function
# f(x) = e^(kx + b)

def Func(v, x):
	if len(v.shape) == 3:
		v = v[:,0,:]
	return torch.exp(v[:,0] * x + v[:,1])

def Error(param, x, y):
	return Func(param, x) - y

k = 5
b = 3

gtParam = torch.from_numpy(np.array([k, b])).view(1,-1).double()

x = torch.from_numpy(np.linspace(0, 1, 100)).double()
indices = torch.from_numpy(np.array([0 for i in range(x.shape[0])])).long()
y = Func(gtParam[indices], x)

plt.plot(x, y, color='green', label='GT Curve')

y = y + (torch.rand(y.shape) * 2 - 1) * 400

plt.plot(x, y, 'b+', label='Data Point')

param = gtParam.clone()
param.zero_()

Solve(variables = [param],
	constants = [x, y],
	indices = [indices],
	fn = Error,
	numIterations = 1000,
	numSuccessIterations = 1000)

print('GT Param = [%f %f]; Estimated Param = [%f %f].'%(
	k, b, param[0,0].item(), param[0,1].item()))

y = Func(param[indices], x)
plt.plot(x, y, color='red', label='Estimation')

plt.legend()
plt.grid()
plt.savefig('curve_fitting.png')