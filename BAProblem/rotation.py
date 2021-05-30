import torch
import math

import numpy as np

def RotationMatrixToQuaternion(R):
  trace = R[0, 0] + R[1, 1] + R[2, 2]
  q = torch.zeros(4, dtype=torch.float64)
  if trace > 0.0:
    t = math.sqrt(trace + 1.0)
    q[0] = 0.5 * t
    t = 0.5 / t
    q[1] = (R[2, 1] - R[1, 2]) * t
    q[2] = (R[0, 2] - R[2, 0]) * t
    q[3] = (R[1, 0] - R[0, 1]) * t
  else:
    i = 0
    if R[1, 1] > R[0, 0]:
      i = 1
    if R[2, 2] > R[i, i]:
      i = 2

    j = (i + 1) % 3
    k = (j + 1) % 3
    t = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1)

    q[i + 1] = 0.5 * t
    t = 0.5 / t
    q[0] = (R[k, j] - R[j, k]) * t
    q[j + 1] = (R[j, i] + R[i, j]) * t
    q[k + 1] = (R[k, i] + R[i, k]) * t
  return q

def QuaternionToAngleAxis(Q):
  q1 = Q[1]
  q2 = Q[2]
  q3 = Q[3]
  sinSquaredTheta = q1 * q1 + q2 * q2 + q3 * q3
  if sinSquaredTheta > 0:
    sinTheta = math.sqrt(sinSquaredTheta)
    cosTheta = Q[0]
    if cosTheta < 0:
      twoTheta = np.arctan2(-sinTheta, -cosTheta) * 2
    else:
      twoTheta = np.arctan2(sinTheta, cosTheta) * 2
    k = twoTheta / sinTheta
  else:
    k = 2
  return torch.from_numpy(np.array([q1 * k, q2 * k, q3 * k]))

def AngleAxisToRotationMatrix(angleAxis):
  kOne = 1.0
  theta2 = torch.sum(angleAxis * angleAxis, dim=1).view(-1, 1)

  mask0 = (theta2 > 0).float()

  theta = torch.sqrt(theta2 + (1 - mask0))
  wx = angleAxis[:,0:1] / theta
  wy = angleAxis[:,1:2] / theta
  wz = angleAxis[:,2:3] / theta
  costheta = torch.cos(theta)
  sintheta = torch.sin(theta)

  R00 =     (costheta   + wx*wx*(kOne -    costheta)) * mask0\
    + kOne * (1 - mask0)
  R10 =  (wz*sintheta   + wx*wy*(kOne -    costheta)) * mask0\
    + angleAxis[:,2:3] * (1 - mask0)
  R20 = (-wy*sintheta   + wx*wz*(kOne -    costheta)) * mask0\
    - angleAxis[:,1:2] * (1 - mask0)
  R01 =  (wx*wy*(kOne - costheta)     - wz*sintheta) * mask0\
    - angleAxis[:,2:3] * (1 - mask0)
  R11 =     (costheta   + wy*wy*(kOne -    costheta)) * mask0\
    + kOne * (1 - mask0)
  R21 =  (wx*sintheta   + wy*wz*(kOne -    costheta)) * mask0\
    + angleAxis[:,0:1] * (1 - mask0)
  R02 =  (wy*sintheta   + wx*wz*(kOne -    costheta)) * mask0\
    + angleAxis[:,1:2] * (1 - mask0)
  R12 = (-wx*sintheta   + wy*wz*(kOne -    costheta)) * mask0\
    - angleAxis[:,0:1] * (1 - mask0)
  R22 =     (costheta   + wz*wz*(kOne -    costheta)) * mask0\
    + kOne * (1 - mask0)

  return torch.cat((R00, R01, R02, R10, R11, R12, R20, R21, R22),\
    dim=1).view(-1, 3, 3)


def RotationMatrixToAngleAxis(R):
  res = torch.zeros(R.shape[0], 3, dtype=torch.float64)
  for i in range(R.shape[0]):
    res[i] = QuaternionToAngleAxis(RotationMatrixToQuaternion(R[i]))
  return res

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


def Normalize(points):
  l = torch.sqrt(torch.sum(points * points, dim=1) + 1e-10)
  l = l.reshape((l.shape[0], 1))
  l = torch.cat([l, l, l], dim=1)
  points_normalized = points / l

  return points_normalized

def EquirectangularProjection(points, width = 5760.0):
  pn = Normalize(points)
  lon = torch.atan2(pn[:,0], pn[:,2])
  hypot = torch.sqrt(pn[:,0]*pn[:,0] + pn[:,2]*pn[:,2])
  lat = torch.atan2(-pn[:,1], hypot)

  x = lon / (2.0 * np.pi)
  mask = (x < 0).float()
  x = mask * (-0.5 - x) + (1 - mask) * (0.5 - x)

  y = lat / (-2.0 * np.pi)

  x = x.reshape((x.shape[0], 1))
  y = y.reshape((y.shape[0], 1))

  return torch.cat([x,y], dim=1) * width