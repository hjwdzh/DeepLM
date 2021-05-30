import numpy as np
import trimesh
import torch
from TorchLM.solver import Solve

from ARAPProblem.io import *

mesh = trimesh.load('data/icp/test1.obj')

V = np.array(mesh.vertices)
F = np.array(mesh.faces)

bboxMin = -1.5
bboxMax = 1.5
bboxRes = 64

x = np.linspace(bboxMin, bboxMax, bboxRes)
stride = x[1] - x[0]

x, y, z = np.meshgrid(x, x, x, indexing='ij')

x = np.reshape(x, (bboxRes ** 3, 1))
y = np.reshape(y, (bboxRes ** 3, 1))
z = np.reshape(z, (bboxRes ** 3, 1))

p = np.concatenate((x,y,z), axis=1)

# consider libigl's point_to_mesh_distance, which is much faster...
_, distance, _ = trimesh.proximity.closest_point(mesh, p)
print('start')

device = 'cuda'

distance = torch.from_numpy(distance).to(device)
V = torch.from_numpy(V).to(device)

def AngleAxisRotatePoint(angleAxis, pt):
  theta2 = (angleAxis * angleAxis).sum(dim=1)

  mask = (theta2 > 0).float()

  theta = torch.sqrt(theta2 + (1 - mask) )

  mask = mask.reshape((mask.shape[0], 1))
  mask = torch.cat([mask, mask, mask], dim=1)

  costheta = torch.cos(theta)
  sintheta = torch.sin(theta)
  thetaInverse = 1.0 / theta

  w0 = angleAxis[:,0] * thetaInverse
  w1 = angleAxis[:,1] * thetaInverse
  w2 = angleAxis[:,2] * thetaInverse

  wCrossPt0 = w1 * pt[:,2] - w2 * pt[:,1]
  wCrossPt1 = w2 * pt[:,0] - w0 * pt[:,2]
  wCrossPt2 = w0 * pt[:,1] - w1 * pt[:,0]

  tmp = (w0 * pt[:,0] + w1 * pt[:,1] + w2 * pt[:,2]) * (1.0 - costheta)

  r0 = pt[:,0] * costheta + wCrossPt0 * sintheta + w0 * tmp
  r1 = pt[:,1] * costheta + wCrossPt1 * sintheta + w1 * tmp
  r2 = pt[:,2] * costheta + wCrossPt2 * sintheta + w2 * tmp

  r0 = r0.reshape((r0.shape[0], 1))
  r1 = r1.reshape((r1.shape[0], 1))
  r2 = r2.reshape((r2.shape[0], 1))
  
  res1 = torch.cat([r0, r1, r2], dim=1)

  wCrossPt0 = angleAxis[:,1] * pt[:,2] - angleAxis[:,2] * pt[:,1]
  wCrossPt1 = angleAxis[:,2] * pt[:,0] - angleAxis[:,0] * pt[:,2]
  wCrossPt2 = angleAxis[:,0] * pt[:,1] - angleAxis[:,1] * pt[:,0]

  r00 = pt[:,0] + wCrossPt0;
  r01 = pt[:,1] + wCrossPt1;
  r02 = pt[:,2] + wCrossPt2;

  r00 = r00.reshape((r00.shape[0], 1))
  r01 = r01.reshape((r01.shape[0], 1))
  r02 = r02.reshape((r02.shape[0], 1))

  res2 = torch.cat([r00, r01, r02], dim=1)

  return res1 * mask + res2 * (1 - mask)

def Distance(transform, vertices):
	global distance, stride
	if len(transform.shape) == 3:
		transform = transform[:,0,:]

	V = AngleAxisRotatePoint(transform[:,:3], vertices) + transform[:,3:6]

	x = (V[:,0] - bboxMin) / stride
	y = (V[:,1] - bboxMin) / stride
	z = (V[:,2] - bboxMin) / stride

	xIdx = x.data.long()
	yIdx = y.data.long()
	zIdx = z.data.long()

	x.data -= xIdx
	y.data -= yIdx
	z.data -= zIdx

	i000 = (xIdx * bboxRes + yIdx) * bboxRes + zIdx
	i001 = i000 + 1
	i010 = i000 + bboxRes
	i011 = i001 + bboxRes
	i100 = i000 + bboxRes * bboxRes
	i101 = i001 + bboxRes * bboxRes
	i110 = i010 + bboxRes * bboxRes
	i111 = i011 + bboxRes * bboxRes

	dis = ((distance[i000] * (1 - z) + distance[i001] * z) * (1 - y)\
		+ (distance[i010] * (1 - z) + distance[i011] * z) * y) * (1 - x)\
		+ ((distance[i100] * (1 - z) + distance[i101] * z) * (1 - y)\
		+ (distance[i110] * (1 - z) + distance[i111] * z) * y) * x

	dis = dis * (dis.data < 0.1).double()
	return dis


refMesh = trimesh.load('data/icp/test2.obj')
Vref = np.array(refMesh.vertices)
Vref = torch.from_numpy(Vref).double().to(device)

R = torch.zeros(1,6).double().to(device)
Idx = torch.zeros(Vref.shape[0]).long().to(device)

Solve(variables = [R],
	constants = [Vref],
	indices = [Idx],
	fn = Distance,
	numIterations = 1000,
	numSuccessIterations = 1000)

R = R[Idx]
Vref = AngleAxisRotatePoint(R[:,:3], Vref) + R[:,3:6]
Vref = Vref.data.cpu().numpy()
Fref = np.array(refMesh.faces)

SaveOBJ('obj/test.obj', Vref, Fref)