import argparse

from ARAPProblem.io import *
from ARAPProblem.loss import *

from TorchLM.solver import SolveFunc, FunctionBlock
from time import time

parser = argparse.ArgumentParser(description='ARAP')
parser.add_argument('--inputFile', default='data/dragon_obj/dragon.OBJ')
parser.add_argument('--outputFile', default='obj/result.obj')
parser.add_argument('--device', default='cuda') #cpu/cuda
args = parser.parse_args()

V, F = LoadOBJ(args.inputFile)
vc = np.mean(V, axis=0)
for i in range(3):
	V[:,i] -= vc[i]

SaveOBJ('obj/origin.obj', V, F)

V = torch.from_numpy(V)
F = torch.from_numpy(F)

VFrames = torch.zeros((V.shape[0], 3)).double()

device = args.device
V, F, VFrames = V.to(device), F.to(device), VFrames.to(device)


# build rigidity
e1Idx = torch.cat((F[:,0], F[:,1], F[:,2]))
e2Idx = torch.cat((F[:,1], F[:,2], F[:,0]))

srcIdx = torch.cat((e1Idx, e2Idx))
tarIdx = torch.cat((e2Idx, e1Idx))

originOffset = V[tarIdx] - V[srcIdx]

vIndices = torch.cat((srcIdx.view(-1,1), tarIdx.view(-1,1)), dim=1)

rigidityFunc = FunctionBlock(variables = [V, VFrames],
	constants = [originOffset],
	indices = [vIndices, srcIdx],
	fn = RigidityError)

# build controls
controlIdx = torch.from_numpy(np.array([
	2,5000,10000,15000,20000,25000,30000])).long()
controlIdx = controlIdx.to(device)
targetPt = V[controlIdx].clone()

py = targetPt[0,2].clone()
theta = 30.0 / 180.0 * np.pi
targetPt[0,0] = np.sin(theta) * py
targetPt[0,2] = np.cos(theta) * py

distanceFunc = FunctionBlock(variables = [V],
	constants = [targetPt],
	indices = [controlIdx],
	fn = DistanceError)

t1 = time()

SolveFunc(funcs = [rigidityFunc, distanceFunc],
	numIterations = 25,
	numSuccessIterations = 25)

t2 = time()

print("Time used %f secs."%(t2 - t1))

V = V.data.cpu().numpy()
F = F.data.cpu().numpy()
SaveOBJ(args.outputFile, V, F)
