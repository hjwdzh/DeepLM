#ifndef TORCHLM_KERNEL_IMPL_H_
#define TORCHLM_KERNEL_IMPL_H_

typedef double T;

void JacobiLeftMultiplyGPU(long* dIndices, T* dResidual, T* dJ, T* dJtr,
	int numResiduals, int numDimP, int numDimJ, int numVar, int varDim, int numP,
	int indicesIdx, int indicesDim);

void JacobiRightMultiplyGPU(T* dP, T* dJ, long* dIndices, T* dResidual,
	int numResiduals, int numDimJ, int numDimP, int numVar, int varDim, int indicesDim);

void ColumnSquareGPU(long* dIndices, T* dColumnNorm, T* dJacobian,
	int numResiduals, int residualDim, int numDimP, int numDimJ, int varDim, int indicesDim);

void ColumnNormInverseGPU(T* dColumnNorm, int num);

void JacobianNormalizeGPU(long* dIndices, T* dColumnNorm, T* dJacobian,
	int residualNum, int residualDim, int numDimP, int numDimJ, int varDim, int indicesDim);

void JacobiBlockJtJGPU(T* dJ, long* dIndices, T* dJtJ,
	int numDimJ, int numResiduals, int numDimP, int numVar, int varDim, int indicesDim);

void JacobiBlockAddDiagonalGPU(T* dJtJ, T* dD, int numV, int numP);

void ListRightMultiplyGPU(T* A, T* X, T* B, int numBlocks, int numDim);

void SquareDotGPU(T* dJtr, T* dD, T* dP, int s);

#endif