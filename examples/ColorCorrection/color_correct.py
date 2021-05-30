import argparse
import os
import cv2
import numpy as np
import torch
from TorchLM.solver import SolveFunc, FunctionBlock

parser = argparse.ArgumentParser(description='Color Correction')
parser.add_argument('--dir', default='data/ColorCorrection')
#parser.add_argument('--src', default='data/landscape.jpg')
parser.add_argument('--device', default='cuda') #cpu/cuda
args = parser.parse_args()

color = cv2.imread(args.src)

stride = 180
width = 256

#os.mkdir('data/ColorCorrection')

for i in range(0, color.shape[0], stride):
	for j in range(0, color.shape[1], stride):
		lx = j
		ly = i
		sy = i + width
		if sy > color.shape[0]:
			ly = color.shape[0] - width
			sy = color.shape[0]
		sx = j + width
		if sx > color.shape[1]:
			sx = color.shape[1]
			lx = color.shape[1] - width

		snap = color[ly:sy, lx:sx,:].copy()
		hsv = cv2.cvtColor(snap, cv2.COLOR_BGR2HSV_FULL)
		snap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

		x = np.random.rand(3) * 2 - 1

		hue = hsv[:,:,0:1] + (x[0] * 6).astype('uint8')
		x[1] *= 30
		x[2] *= 60
		if x[1] + np.min(hsv[:,:,1:2]) < 0:
			x[1] = -np.min(hsv[:,:,1:2])
		if x[1] + np.max(hsv[:,:,1:2]) > 255:
			x[1] = 255 - np.max(hsv[:,:,1:2])

		if x[2] + np.min(hsv[:,:,2:3]) < 0:
			x[2] = -np.min(hsv[:,:,2:3])
		if x[2] + np.max(hsv[:,:,2:3]) > 255:
			x[2] = 255 - np.max(hsv[:,:,2:3])

		sat = hsv[:,:,1:2].astype('float32') + x[1] * 30
		val = hsv[:,:,2:3].astype('float32') + x[2] * 60

		sat = np.clip(sat, 0, 255).astype('uint8')
		val = np.clip(val, 0, 255).astype('uint8')

		hsv = np.concatenate([hue, sat, val], axis=2).astype('uint8')

		snap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

		cv2.imwrite('data/ColorCorrection/snap_%03d_%03d.jpg'%(ly, lx), snap)

imageFiles = [f for f in os.listdir(args.dir) if f[-3:] == 'jpg']

offsets = []
images = []
sizeX = 0
sizeY = 0
for fn in imageFiles:
	im = cv2.imread(args.dir + '/' + fn)
	w = fn[:-4].split('_')
	py = int(w[1])
	px = int(w[2])
	images.append(im)
	offsets.append([py, px])
	if py + im.shape[0] > sizeY:
		sizeY = py + im.shape[0]
	if px + im.shape[1] > sizeX:
		sizeX = px + im.shape[1]


result = np.zeros((sizeY, sizeX, 3))
weights = np.zeros((sizeY, sizeX))

for i in range(len(images)):
	ly = offsets[i][0]
	lx = offsets[i][1]
	ry = ly + images[i].shape[0]
	rx = lx + images[i].shape[1]

	weights[ly:ry, lx:rx] += 1
	result[ly:ry, lx:rx, :] += images[i]

# get initial guess for the results
for i in range(3):
	result[:,:,i] /= weights

for i in range(len(images)):
	ly = offsets[i][0]
	lx = offsets[i][1]
	ry = ly + images[i].shape[0]
	rx = lx + images[i].shape[1]

	Y = np.linspace(ly, ry - 1, (ry - ly))
	X = np.linspace(lx, rx - 1, (rx - lx))

	X = X.astype('int32')
	Y = Y.astype('int32')
	X, Y = np.meshgrid(X, Y)

	pixIdx = torch.from_numpy(Y * sizeX + X).view(-1).long()
	colorIdx = torch.ones(pixIdx.shape[0]).long() * i
	pixVal = torch.from_numpy(images[i]).view(-1, 3).double()

	if i == 0:
		pixIndices = pixIdx
		colorIndices = colorIdx
		pixValues = pixVal
	else:
		pixIndices = torch.cat((pixIndices, pixIdx))
		colorIndices = torch.cat((colorIndices, colorIdx))
		pixValues = torch.cat((pixValues, pixVal), dim=0)

targetValues = torch.from_numpy(result).view(-1, 3).double()

def BSplineTransform(knots, pix):
	#Nx3x8
	#Nx3
	s = pix / 256.0
	idx = (s * 5).data.long()
	idx = torch.clamp(idx, 0, 4)

	x = s * 5 - idx

	idx = idx.view(idx.shape[0], idx.shape[1], 1)
	k1 = torch.gather(knots, 2, idx)[:,:,0]
	k2 = torch.gather(knots, 2, idx + 1)[:,:,0]
	k3 = torch.gather(knots, 2, idx + 2)[:,:,0]
	k4 = torch.gather(knots, 2, idx + 3)[:,:,0]

	b1 = (1 - x)**3 / 6.0
	b2 = (3 * (x**3) - 6 * x * x + 4) / 6.0
	b3 = (-3 * (x**3) + 3 * x * x + 3 * x + 1) / 6.0
	b4 = (x ** 3) / 6.0

	return (b1 * k1 +  b2 * k2 + b3 * k3 + b4 * k4) * 256


def ColorDiff(target, knots, pixValues):
	target = target[:,0,:]
	knots = knots[:,0,:,:]

	res = BSplineTransform(knots, target)

	return res - pixValues

device = args.device
targetValues = targetValues.to(device)
pixValues = pixValues.to(device)
pixIndices = pixIndices.to(device)
colorIndices = colorIndices.to(device)

knots = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
knots = [knots for i in range(3)]
knots = [knots for i in range(len(images))]
knots = torch.from_numpy(np.array(knots)).double().to(device)

'''
dataFunc = FunctionBlock(variables = [targetValues, knots],
	constants = [pixValues],
	indices = [pixIndices, colorIndices],
	fn = ColorDiff)

SolveFunc(funcs = [dataFunc],
	numIterations = 30,
	numSuccessIterations = 30)
'''
targetValues = targetValues.view(sizeY, sizeX, 3).data.cpu().numpy()

targetValues = np.clip(targetValues, 0, 255).astype('uint8')

cv2.imwrite('obj/result.png', targetValues)