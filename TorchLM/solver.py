import math

from .jacobian import *
from .listvec import *
import LMCore
import subprocess
from time import time

class Strategy:
	def __init__(self):
		self.decreaseFactor = 2
		self.radius = 1e4

	def reject(self):
		self.radius /= self.decreaseFactor
		self.decreaseFactor *= 2

	def accept(self, quality):
		self.radius /= max(1.0 / 3.0,
			1.0 - pow(2.0 * quality - 1.0, 3))
		self.decreaseFactor = 2.0


class StepEvaluator:
	def __init__(self, referenceCost):
		self.referenceCost = referenceCost
		self.minimumCost = referenceCost;
		self.currentCost = referenceCost;
		self.candidateCost = referenceCost;

		self.accumulatedReferenceModelCostChange = 0
		self.accumulatedCandidateModelCostChange = 0
		self.numConsecutiveNonmonotonicSteps = 0
		self.maxConsecutiveNonmonotonicSteps = 0

	def accept(self, cost, modelCostChange):
		self.currentCost = cost
		self.accumulatedCandidateModelCostChange += modelCostChange
		self.accumulatedReferenceModelCostChange += modelCostChange
		if self.currentCost < self.minimumCost:
			self.minimumCost = self.currentCost
			self.numConsecutiveNonmonotonicSteps = 0
			self.candidateCost = self.currentCost
			self.accumulatedCandidateModelCostChange = 0
		else:
			self.numConsecutiveNonmonotonicSteps += 1
			if self.currentCost > self.candidateCost:
				self.candidateCost = self.currentCost
				self.accumulatedCandidateModelCostChange = 0

		if self.numConsecutiveNonmonotonicSteps ==\
			self.maxConsecutiveNonmonotonicSteps:
			self.referenceCost = self.candidateCost
			self.accumulatedReferenceModelCostChange =\
				self.accumulatedCandidateModelCostChange

	def StepQuality(self, cost, modelCostChange):
		relativeDecrease = (self.currentCost - cost) / modelCostChange
		historicalRelativeDecrease = (self.referenceCost - cost)\
			/ (self.accumulatedReferenceModelCostChange + modelCostChange)

		return max(relativeDecrease, historicalRelativeDecrease)

class Summary:
	def __init__(self):
		self.cost = 0
		self.gradientNorm = 0
		self.gradientMaxNorm = 0
		self.cost = None
		self.lmIteration = 0
		self.linearIteration = 0
		self.numConsecutiveInvalidSteps = 0
		self.stepIsValid = True
		self.relativeDecrease = 0
		# 0 = no-converge, 1 = success, 2 = fail
		self.linearTerminationType = 0
		self.lmTerminationType = 0

class LMOption:
	def __init__(self):
		self.minDiagonal = 1e-6
		self.maxDiagonal = 1e32
		self.eta = 1e-3
		self.residualResetPeriod = 10
		self.maxLinearIterations = 150
		self.maxNumIterations = 16
		self.maxSuccessIterations = 16
		self.maxInvalidStep = 5
		self.minRelativeDecrease = 1e-3
		self.parameterTolerance = 1e-8
		self.functionTolerance = 1e-6
		self.radius = 1e4
		self.span = 4000000

class FunctionBlock:
	def __init__(self, variables=[], constants=[], indices = [], fn=None):
		self.variables = variables
		self.constants = constants
		self.indices = indices
		self.fn = fn

class LMSolver:
	def __init__(self, functions, verbose = True, option = LMOption()):
		torch.cuda.synchronize()
		self.startTime = time()
		self.verbose = verbose

		self.functions = functions
		self.variableDict = {}
		self.variables = []
		self.vrangeFunc = []
		for func in functions:
			for v in func.variables:
				if v in self.variableDict:
					continue
				self.variableDict[v] = len(self.variables)
				self.variables.append(v)
			self.vrangeFunc.append(range(len(func.variables)))
		self.vranges = range(len(self.variables))

		self.vidsFunc = []
		for func in functions:
			self.vidsFunc.append([self.variableDict[v] for v in func.variables])

		self.summary = Summary()
		self.strategy = Strategy()
		self.strategy.radius = option.radius
		self.evaluator = StepEvaluator(0)
		self.option = option

		self.qTolerance = 0.0
		self.rTolerance = 0.0
		self.xNorm = -1
		self.modelCostChange = 0.0
		self.delta = None
		
		self.q = None
		self.z = None
		self.p = None
		self.r = None
		self.xref = None
		self.bref = None
		self.modelResiduals = None
		self.gradients = None
		self.jacobians = None
		self.preconditioner = None

		self.isCuda = self.variables[0].is_cuda

	def MemoryTensor(self,b):
		k = 8
		for i in range(len(b.shape)):
			k *= b.shape[i]
		return k

	def MemoryList(self,b):
		mem = 0
		for l in b:
			if isinstance(l, list):
				mem += self.MemoryList(l)
			elif torch.is_tensor(l):
				mem += self.MemoryTensor(l)
		return mem

	def Memory(self):
		mem = 0
		for a,b in self.__dict__.items():
			if isinstance(b, list):
				mem += self.MemoryList(b)
			if torch.is_tensor(b):
				mem += self.MemoryTensor(b)
		return mem


	def Timing(self):
		if self.isCuda:
			torch.cuda.synchronize()

		return time()

	def InitializeVariables(self):
		self.bref = ListZero(self.variables)
		self.xref = ListZero(self.variables)
		self.r = ListZero(self.variables)
		self.z = ListZero(self.variables)
		self.p = ListZero(self.variables)
		self.q = ListZero(self.variables)
		self.gradients = ListZero(self.variables)
		self.jacobianScale = ListZero(self.variables)
		self.diagonal = ListZero(self.variables)
		self.preconditioner = []
		for v in self.variables:
			l = 1
			for j in range(1, len(v.shape)):
				l *= v.shape[j]
			self.preconditioner.append(torch.zeros((v.shape[0], l, l),
				dtype=v.dtype, device=v.device))

	def EvaluateCost(self, candidate):
		span = self.option.span
		cost = 0

		for funcId in range(len(self.functions)):
			func = self.functions[funcId]
			indices = func.indices
			variables = [candidate[j] for j in self.vidsFunc[funcId]]
			constants = func.constants
			residualNum = indices[0].shape[0]
			for dim in range(0, residualNum, span):
				start = dim
				end = start + span
				if end > residualNum:
					end = residualNum

				varIndexed = [variables[i][indices[i][start:end]]\
					for i in range(len(variables))]
				constantsPar = [constants[i][start:end]\
					for i in range(len(constants))]

				residuals = func.fn(*varIndexed, *constantsPar)

				cost += torch.sum(0.5 * residuals * residuals).item()
				del varIndexed, constantsPar, residuals
				
				if self.isCuda:
					torch.cuda.empty_cache()
		
		return cost

	def Evaluate(self, isFirstTime=False):
		# compute gradients twice, might be a bit inefficient
		for i in self.vranges:
			self.variables[i].grad = None
			self.gradients[i].zero_()
			self.jacobianScale[i].zero_()

		self.cost = 0

		for funcId in range(len(self.functions)):
			indices = self.functions[funcId].indices
			variables = self.functions[funcId].variables
			constants = self.functions[funcId].constants
			fn = self.functions[funcId].fn
			vrange = self.vrangeFunc[funcId]

			residualNum = indices[0].shape[0]
			# test residual
			if isFirstTime:
				varTemp = [variables[i][indices[i][:1]] for i in vrange]
				constantTemp = [constants[i][:1]\
					for i in range(len(constants))]
				residualsTemp = fn(*varTemp, *constantTemp)
				residualsTemp = residualsTemp.view(residualsTemp.shape[0], -1)

				residualDim = residualsTemp.shape[1]

				self.functions[funcId].jacobians = []
				maxDim = 0
				for i in vrange:
					v = variables[i]
					v = v.view(v.shape[0], -1)
					jacobian = torch.zeros(residualDim, *indices[i].shape,
						*v.shape[1:], device=v.device, dtype=v.dtype)
					if v.shape[1] > maxDim:
						maxDim = v.shape[1]
					self.functions[funcId].jacobians.append(jacobian)

				self.functions[funcId].buffer = torch.zeros(indices[0].shape[0],
					maxDim, device=indices[0].device, dtype=variables[0].dtype)

				self.functions[funcId].residuals = torch.zeros(residualNum,
					residualDim, device=indices[0].device,
					dtype=variables[0].dtype)

			span = self.option.span

			for dim in range(0, residualNum, span):
				start = dim
				end = start + span
				if end > residualNum:
					end = residualNum
				varIndexed = [variables[i][indices[i][start:end]]\
					for i in vrange]
				varIndexed = [torch.nn.Parameter(v) for v in varIndexed]
				constantsPar = [constants[i][start:end]\
					for i in range(len(constants))]

				residuals = fn(*varIndexed, *constantsPar)
				residuals = residuals.view(residuals.shape[0], -1)

				cost = torch.sum(0.5 * residuals * residuals)
				# collect gradients and jacobians

				for i in range(residuals.shape[1]):
					for j in vrange:
						varIndexed[j].grad = None

					l = torch.sum(residuals[:,i])
					l.backward(retain_graph = True)

					for j in vrange:
						grad = varIndexed[j].grad
						if grad is None:
							self.functions[funcId].\
								jacobians[j][i][start:end].zero_()
							continue
						grad = grad.data
						grad = grad.view(grad.shape[0], grad.shape[1], -1)
						self.functions[funcId].jacobians[j][i][start:end].copy_(
							grad)

					del l

				self.functions[funcId].residuals[start:end].copy_(
					residuals.data)
				self.cost += cost.item()
				del varIndexed, constantsPar, residuals, cost
		
				if self.isCuda:
					torch.cuda.empty_cache()


			residuals = self.functions[funcId].residuals

			gradients = [self.gradients[i] for i in self.vidsFunc[funcId]]
			jacobians = self.functions[funcId].jacobians

			#if self.isCuda:
			#	JacobiLeftMultiply(jacobians, residuals, self.variables, indices,
			#		gradients)
			#else:
			LMCore.JacobiLeftMultiply(jacobians, residuals, indices,
				self.functions[funcId].buffer, gradients, 0)

			#self.jacobianScale = JacobiNormalize(jacobians, indices,
			#	self.variables)

			jacobianScale = [self.jacobianScale[i]\
				for i in self.vidsFunc[funcId]]
			LMCore.JacobiColumnSquare(indices, jacobians, jacobianScale, 0)

		LMCore.ColumnInverseSquare(jacobianScale)

		torch.cuda.synchronize()
		for funcId in range(len(self.functions)):
			indices = self.functions[funcId].indices
			jacobians = self.functions[funcId].jacobians
			jacobianScale = [self.jacobianScale[i]\
				for i in self.vidsFunc[funcId]]

			LMCore.JacobiNormalize(indices, jacobianScale, jacobians)

		self.summary.gradientNorm = ListNorm(self.gradients)
		self.summary.gradientMaxNorm = ListMaxNorm(self.gradients)

		if self.isCuda:
			torch.cuda.empty_cache()

	def JacobiJtJD(self, diagonal, p, residuals, jtrs):
		for jtr in jtrs:
			jtr.zero_()

		# jacobirightmultiply
		for i in range(len(self.functions)):
			func = self.functions[i]
			jacobians = func.jacobians
			pTemp = [p[j] for j in self.vidsFunc[i]]
			indices = func.indices
			LMCore.JacobiRightMultiply(jacobians, pTemp, indices, residuals[i])

		# jacobileftmultiply
		for i in range(len(self.functions)):
			func = self.functions[i]
			jacobians = func.jacobians
			indices = func.indices
			buf = func.buffer
			jtrsTemp = [jtrs[j] for j in self.vidsFunc[i]]
			LMCore.JacobiLeftMultiply(jacobians, residuals[i], indices, buf,
				jtrsTemp, 0)

		LMCore.SquareDot(diagonal, p, jtrs)

	def LinearSolve(self, lmDiagonal):
		xref = self.xref
		bref = self.bref
		r = self.r
		z = self.z
		p = self.p
		q = self.q

		#if self.isCuda:
		#	JacobiLeftMultiply(self.jacobians, self.residuals, self.variables,
		#		indices, bref)
		#else:

		for b in bref:
			b.zero_()

		for i in range(len(self.functions)):
			jacobians = self.functions[i].jacobians
			residuals = self.functions[i].residuals

			indices = self.functions[i].indices
			buf = self.functions[i].buffer
			brefTemp = [bref[j] for j in self.vidsFunc[i]]
			LMCore.JacobiLeftMultiply(jacobians, residuals, indices,
				buf, brefTemp, 0)

		#if self.isCuda:
		#	JacobiBlockJtJ(self.jacobians, lmDiagonal, self.variables, indices,
		#		self.preconditioner)			
		#else:

		for i in range(len(self.preconditioner)):
			self.preconditioner[i].zero_()

		for i in range(len(self.functions)):
			jacobians = self.functions[i].jacobians
			indices = self.functions[i].indices

			lmDiagonalTemp = [lmDiagonal[j] for j in self.vidsFunc[i]]
			preconditioner = [self.preconditioner[j] for j in self.vidsFunc[i]]

			LMCore.JacobiBlockJtJ(jacobians, lmDiagonalTemp,
				indices, preconditioner, 0)

		preconditioner = self.preconditioner
		ListInvert(preconditioner)

		self.summary.linearTerminationType = 0
		self.summary.lmIteration = 0

		normB = ListNorm(bref)

		for i in self.vranges:
			xref[i].zero_()

		if normB == 0:
			self.summary.linearTerminationType = 1
			return xref

		tolR = self.rTolerance * normB
		for i in self.vranges:
			r[i].copy_(bref[i])

		rho = 1.0
		Q0 = -0.0

		self.summary.numIterations = 0

		d0 = 0
		d1 = 0
		d2 = 0
		while (True):
			#t1 = self.Timing()
			self.summary.numIterations += 1

			#if self.isCuda:
			#	ListRightMultiply(preconditioner, r, self.z)
			#else:
			LMCore.ListRightMultiply(preconditioner, r, self.z)

			lastRho = rho
			rho = ListDot(r, z)

			if self.summary.numIterations == 1:
				for i in self.vranges:
					p[i].copy_(z[i])
			else:
				beta = rho / lastRho
				for i in self.vranges:
					p[i] *= beta
					p[i] += z[i]
			
			#t2 = self.Timing()
			#if self.isCuda:
			#	JacobiJtJD(self.jacobians, lmDiagonal, p, self.variables,
			#		indices, self.modelResiduals, self.q)
			#else:
			self.JacobiJtJD(lmDiagonal, p, self.modelResiduals, q)

			#t3 = self.Timing()
			pq = ListDot(p, q)
			if pq < 0:
				self.summary.linearTerminationType = 2
				break
				#return xref

			alpha = rho / pq
			for i in self.vranges:
				xref[i] += alpha * p[i]

			# this is to avoid numercial issue: recompute r every reset steps
			if self.summary.numIterations % \
				self.option.residualResetPeriod == 0:
				#if self.isCuda:
				#	JacobiJtJD(self.jacobians, lmDiagonal, xref, self.variables,
				#		indices, self.modelResiduals, self.q)
				#else:

				self.JacobiJtJD(lmDiagonal, xref, self.modelResiduals, q)
				for i in self.vranges:
					r[i].copy_(bref[i] - self.q[i])
				r = [bref[i] - self.q[i] for i in range(len(r))]
			else:
				for i in self.vranges:
					r[i] -= alpha * q[i]

			Q1 = -1.0 * ListDot(xref, [bref[i] + r[i] for i in range(len(r))])
			zeta = self.summary.numIterations * (Q1 - Q0) / Q1

			if zeta < self.qTolerance:
				self.summary.linearTerminationType = 1
				break

			Q0 = Q1
			normR = ListNorm(r)

			if normR < tolR:
				self.summary.linearTerminationType = 1
				break

			if self.summary.numIterations > self.option.maxLinearIterations:
				break
			if self.isCuda:
				torch.cuda.empty_cache()

			#t4 = self.Timing()
	
		if self.isCuda:
			torch.cuda.empty_cache()

		return xref

	def ComputeTrustRegionStep(self):
		#diagonal = JacobiSquaredColumnNorm(self.jacobians,
		#	indices, self.variables)

		t0 = self.Timing()
		for i in range(len(self.diagonal)):
			self.diagonal[i].zero_()

		for i in range(len(self.functions)):
			indices = self.functions[i].indices
			jacobians = self.functions[i].jacobians
			diagonal = [self.diagonal[j] for j in self.vidsFunc[i]]
			LMCore.JacobiColumnSquare(indices, jacobians, diagonal, 0)

		diagonal = self.diagonal

		ListClamp(diagonal, self.option.minDiagonal, self.option.maxDiagonal)

		lmDiagonal = []
		for v in diagonal:
			lmDiagonal.append(torch.sqrt(v / self.strategy.radius))

		self.qTolerance = self.option.eta
		self.rTolerance = -1.0

		step = self.LinearSolve(lmDiagonal);

		for i in self.vranges:
			step[i].copy_(-step[i])

		#if self.isCuda:
		#	JacobiRightMultiply(self.jacobians, step, self.variables, indices,
		#		modelResiduals)
		#else:

		modelResiduals = self.modelResiduals
		for i in range(len(self.functions)):
			indices = self.functions[i].indices
			jacobians = self.functions[i].jacobians
			stepTemp = [step[j] for j in self.vidsFunc[i]]
			LMCore.JacobiRightMultiply(jacobians, stepTemp, indices,
				modelResiduals[i])

		self.modelCostChange = 0

		for i in range(len(self.functions)):
			self.modelCostChange += -torch.sum(modelResiduals[i]
			* (self.functions[i].residuals + modelResiduals[i] * 0.5))

		self.summary.stepIsValid = (self.modelCostChange > 0.0)
		if self.summary.stepIsValid:
			self.delta = [step[i] *
				self.jacobianScale[i] for i in range(len(step))]
			self.summary.numConsecutiveInvalidSteps = 0
		t1 = self.Timing()

	def Solve(self):
		self.InitializeVariables()
		#self.IndicesIdx, self.EndIdx = PrepareSortedIndices(
		#	indices, self.variables)

		self.Evaluate(True)
		if self.option.maxSuccessIterations == 0:
			return
		if self.verbose:
			print('Initial cost = %E, Memory = %E G'%(self.cost,
				self.Memory() / 1024.0 / 1024.0 / 1024.0))

		self.modelResiduals = [self.functions[funcId].residuals.clone()\
			for funcId in range(len(self.functions))]

		if self.summary.lmIteration == 0:
			self.evaluator = StepEvaluator(self.cost)
		
		outerIterations = 0
		successIterations = 0
		self.debug = False
		while (True):
			t1 = self.Timing()
			outerIterations += 1
			if outerIterations == self.option.maxNumIterations:
				break
			self.ComputeTrustRegionStep()

			t2 = self.Timing()
			if not self.summary.stepIsValid:
				self.summary.numConsecutiveInvalidSteps += 1
				if self.summary.numConsecutiveInvalidSteps\
					> self.option.maxInvalidStep:
					self.summary.lmTerminationType = 2
					return
				self.strategy.reject()
				continue

			candidateX = [self.variables[i].data
				+ self.delta[i].reshape(self.variables[i].shape)
				for i in self.vranges]

			cost = self.EvaluateCost(candidateX)

			# parameter tolerance check
			stepSizeTolerance = self.option.parameterTolerance\
				* (self.xNorm + self.option.parameterTolerance)
			stepNorm = ListNorm(self.delta)

			if stepNorm < stepSizeTolerance:
				self.summary.lmTerminationType = 1
				return

			# function tolerance check
			costChange = self.cost - cost
			absoluteFunctionTolerance =\
				self.option.functionTolerance * self.cost

			if abs(costChange) < absoluteFunctionTolerance:
				self.summary.lmTerminationType = 1
				return

			# evaluate relative decrease
			self.summary.relativeDecrease = self.evaluator.StepQuality(
				cost, self.modelCostChange)

			if self.summary.relativeDecrease\
				> self.option.minRelativeDecrease:
				for i in self.vranges:
					self.variables[i].data += self.delta[i].reshape(
						self.variables[i].shape)

				self.xNorm = ListNorm(self.variables).item()

				self.strategy.accept(self.summary.relativeDecrease)
				self.evaluator.accept(cost, self.modelCostChange)

				if self.verbose:
					torch.cuda.synchronize()
					currentTime = time()
					print('iter = %d, cost = %E, radius = %E, CGstep = %d, time = %f'%(
						outerIterations,
						cost,
						self.strategy.radius,
						self.summary.numIterations,
						currentTime - self.startTime))

				#if math.sqrt((self.cost-cost) / indices[0].shape[0]) < 5e-2:
				#	self.cost = cost
				#	break

				successIterations += 1
				if successIterations >= self.option.maxSuccessIterations:
					self.cost = cost
					break

				# backward is slow on cpu...
				self.Evaluate(True)
			else:
				self.strategy.reject()
				if self.verbose:
					print('iter = %d (rejected)'%(outerIterations))

			if self.strategy.radius < 1e-32 or\
				self.summary.gradientMaxNorm < 1e-10:
				self.summary.lmTerminationType = 1
				return

			if self.isCuda:
				torch.cuda.empty_cache()
			#t3 = self.Timing()


def Solve(variables, constants, indices, fn,
	numIterations = 15, numSuccessIterations = 15,
	maxLinearIterations=150, verbose = True):
	if len(indices) == 0:
		return
	for i in range(len(indices)):
		if indices[i].shape[0] == 0:
			return

	func = FunctionBlock(variables=variables, constants=constants,
		indices=indices, fn=fn)

	option = LMOption()
	option.maxLinearIterations = maxLinearIterations
	solver = LMSolver(functions=[func], verbose=verbose, option = option)
	solver.option.maxNumIterations = numIterations
	solver.option.maxSuccessIterations = numSuccessIterations
	for i in range(len(indices)):
		index = indices[i].contiguous()
		if len(index.shape) == 1:
			index = index.view(-1, 1)
		indices[i] = index
	solver.Solve()

	return solver.cost

def SolveFunc(funcs, numIterations = 15,
	numSuccessIterations = 15,
	maxLinearIterations=150, verbose = True):
	for func in funcs:
		indices = func.indices
		for i in range(len(indices)):
			index = indices[i].contiguous()
			if len(index.shape) == 1:
				index = index.view(-1, 1)
			indices[i] = index
	option = LMOption()
	option.maxLinearIterations = maxLinearIterations
	solver = LMSolver(functions=funcs, verbose=verbose, option = option)
	solver.option.maxNumIterations = numIterations
	solver.option.maxSuccessIterations = numSuccessIterations
	solver.Solve()

	return solver.cost
