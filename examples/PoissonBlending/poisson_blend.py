import cv2
import argparse
import torch
import numpy as np

from TorchLM.solver import SolveFunc, FunctionBlock
from time import time

parser = argparse.ArgumentParser(description='Poisson Blending')
parser.add_argument('--src', default='data/blend/target1.jpg')
parser.add_argument('--ref', default='data/blend/source1.jpg')
parser.add_argument('--mask', default='data/blend/mask1.png')
parser.add_argument('--tar', default='obj/tar.png')
parser.add_argument('--device', default='cuda') #cpu/cuda
args = parser.parse_args()

# prepare data
src = cv2.imread(args.src)
ref = cv2.imread(args.ref)
mask = cv2.imread(args.mask)

boundary = cv2.dilate(mask, np.ones((3,3),'uint8')) - mask

sx = ref.shape[1]
sy = ref.shape[0]

offX = 0
offY = mask.shape[0] - 388

mask = mask > 0
mask = mask[-388:,:,0]
boundary = boundary > 0
boundary = boundary[-388:,:,0]

def BackgroundLoss(pix, tar):
	if len(pix.shape) == 3:
		pix = pix[:,0,:]
	return (pix - tar) * 100

def ForegroundLoss(pix, laplacian):
	return 4 * pix[:,0,:] - pix[:,1,:] - pix[:,2,:] - pix[:,3,:]\
		- pix[:,4,:] - laplacian

# construct indices
def ConvertIdx(b, sx):
	return torch.from_numpy(b[:,0] * sx + b[:,1]).view(-1, 1).long()

background = np.argwhere(boundary > 0)
foreground = np.argwhere(mask > 0)

bIdx = ConvertIdx(background, sx)
fIdx = ConvertIdx(foreground, sx)

leftPixels = foreground.copy()
leftPixels[:,1] -= 1

upPixels = foreground.copy()
upPixels[:,0] -= 1

rightPixels = foreground.copy()
rightPixels[:,1] += 1

downPixels = foreground.copy()
downPixels[:,0] += 1

leftPixels = ConvertIdx(leftPixels, sx)
upPixels = ConvertIdx(upPixels, sx)
rightPixels = ConvertIdx(rightPixels, sx)
downPixels = ConvertIdx(downPixels, sx)

pix = torch.from_numpy(ref).double().view(-1, 3)


laplacian = (4 * pix[fIdx] - pix[leftPixels] - pix[upPixels]
	- pix[rightPixels] - pix[downPixels])[:,0,:]

#laplacian.zero_()
fIdx = torch.cat((fIdx, leftPixels, upPixels, rightPixels, downPixels), dim=1)

background = torch.from_numpy(src[offY:offY+sy,offX:offX+sx,:]).double()
background = background.view(-1,3)[bIdx][:,0,:]

d = args.device
pix, background, bIdx, fIdx, laplacian = pix.to(d), background.to(d),\
	bIdx.to(d), fIdx.to(d), laplacian.to(d)

bFunc = FunctionBlock(variables = [pix],
	constants = [background],
	indices = [bIdx],
	fn = BackgroundLoss)

fFunc = FunctionBlock(variables = [pix],
	constants = [laplacian],
	indices = [fIdx],
	fn = ForegroundLoss)

SolveFunc(funcs = [bFunc, fFunc],
	numIterations = 15,
	numSuccessIterations = 15)

pix = pix.view(sy, sx, 3).data.cpu().numpy()
pix = np.clip(pix, 0, 255)

for c in range(3):
	src[offY:offY+sy, offX:offX+sx, c] = pix[:,:,c] * mask +\
		src[offY:offY+sy, offX:offX+sx, c] * (1 - mask)

cv2.imwrite(args.tar, src)