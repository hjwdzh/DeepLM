import torch
import numpy as np
from BAProblem.rotation import RotationMatrixToAngleAxis
import BACore

def LoadBALFromFile(filename):
	return BACore.LoadBALFromFile(filename, 2, 9, 3)

def LoadColmapFromFile(filename):
	return BACore.LoadBALFromFile(filename, 2, 12, 3)

def LoadRiemannData2D(camFile, txtFile, ptbFile):
	points, cameras, features2d, features3d, ptIdx, camIdx, originPtIdx,\
		originCamIdx =BACore.LoadRiemannFromFile(camFile, txtFile, ptbFile)

	R = cameras[:, :9].view(-1, 3, 3)
	T = cameras[:, 9:]
	angleAxis = RotationMatrixToAngleAxis(R)
	return points, torch.cat((angleAxis, T), dim=1), features2d, ptIdx, camIdx

def LoadRiemannDataLarge(eopFile, ptbFile):

	points, cameras, features2d, ptIdx, camIdx\
		= BACore.LoadRiemannLargeFromFile(eopFile, ptbFile)

	R = cameras[:, :9].view(-1, 3, 3)
	T = cameras[:, 9:]
	angleAxis = RotationMatrixToAngleAxis(R)
	return points, torch.cat((angleAxis, T), dim=1), features2d, ptIdx, camIdx