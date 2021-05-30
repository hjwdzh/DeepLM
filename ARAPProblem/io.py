import numpy as np

def LoadOBJ(filename):
	lines = [l.strip() for l in open(filename)]
	vertices = []
	faces = []
	for line in lines:
		words = [w for w in line.split(' ') if w != '']
		if len(words) == 0:
			continue
		if words[0] == 'v':
			vertices.append([float(words[1]), float(words[2]), float(words[3])])
		if words[0] == 'f':
			faces.append([(int(words[i].split('/')[0])-1) for i in [1,2,3]])

	return np.array(vertices, dtype='float64'), np.array(faces, dtype='int64')

def SaveOBJ(filename, V, F):
	fp = open(filename, 'w')
	for i in range(V.shape[0]):
		v = V[i]
		fp.write('v %.6f %.6f %.6f\n'%(v[0], v[1], v[2]))
	for i in range(F.shape[0]):
		v = F[i] + 1
		fp.write('f %d %d %d\n'%(v[0], v[1], v[2]))
	fp.close()
