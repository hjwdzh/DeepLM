import torch
from .rotation import *

def DistanceError(src, tar):
	if len(src.shape) == 3:
		src = src[:,0,:]
	return (src - tar)

def RigidityError(vertices, srcFrames, originOffset):
	srcPt = vertices[:,0,:]
	tarPt = vertices[:,1,:]
	currentOffset = tarPt - srcPt

	if len(srcFrames.shape) == 3:
		srcFrames = srcFrames[:,0,:]
	targetOffset = AngleAxisRotatePoint(srcFrames, originOffset)
	res = (currentOffset - targetOffset)
	return  res

def LinearError(vertices, originOffset):
	srcPt = vertices[:,0,:]
	tarPt = vertices[:,1,:]
	currentOffset = tarPt - srcPt

	return currentOffset - originOffset	