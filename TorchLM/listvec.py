import torch

def ListNorm(listvec):
	n = 0
	for vec in listvec:
		n += torch.sum(vec * vec)
	return torch.sqrt(n)

def ListMaxNorm(listvec):
	n = 0
	for vec in listvec:
		if vec.view(-1).shape[0] == 0:
			continue
		m = torch.max(torch.abs(vec))
		if m > n:
			n = m
	return n

def ListClamp(listvec, minVal, maxVal):
	for i in range(len(listvec)):
		listvec[i] = torch.clamp(listvec[i], min = minVal, max = maxVal)

def ListInvert(listvec):
	for i in range(len(listvec)):
		listvec[i] = torch.inverse(listvec[i])
	torch.cuda.empty_cache()


def ListZero(variables):
	zeros = []
	for v in variables:
		vplain = v.view(v.shape[0], -1)
		zeros.append(torch.zeros(vplain.shape, dtype=v.dtype, device=v.device))
	return zeros

def ListCopy(variables):
	return [v.clone() for v in variables]

def ListRightMultiply(m, r, res):
	#result = []
	for varid in range(len(r)):
		res[varid].copy_(torch.matmul(m[varid],
			r[varid].view(r[varid].shape[0], -1, 1))
			.view(r[varid].shape))
	#return result

def ListDot(a, b):
	res = 0
	for i in range(len(a)):
		res += torch.sum(a[i] * b[i])
	return res