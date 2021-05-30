import argparse
import os
import sys
import torch
import BACore
import numpy as np

from BAProblem.rotation import AngleAxisRotatePoint
from BAProblem.loss import SnavelyReprojectionError
from BAProblem.io import LoadBALFromFile
from TorchLM.solver import Solve

from time import time

parser = argparse.ArgumentParser(description='Bundle adjuster')
parser.add_argument('--balFile', default='data/problem-1723-156502-pre.txt')
parser.add_argument('--device', default='cuda') #cpu/cuda
args = parser.parse_args()

filename = args.balFile
device = args.device

# Load BA data
points, cameras, features, ptIdx, camIdx = LoadBALFromFile(filename)

# Optionally use CUDA
points, cameras, features, ptIdx, camIdx = points.to(device),\
	cameras.to(device), features.to(device), ptIdx.to(device), camIdx.to(device)

if device == 'cuda':
	torch.cuda.synchronize()

t1 = time()
# optimize
Solve(variables = [points, cameras],
	constants = [features],
	indices = [ptIdx, camIdx],
	fn = SnavelyReprojectionError,
	numIterations = 15,
	numSuccessIterations = 15)
t2 = time()

print("Time used %f secs."%(t2 - t1))

