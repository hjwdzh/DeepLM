#include "kernel_impl.h"

#include <iostream>

template <typename T>
struct AtomicFPOp;

template <>
struct AtomicFPOp<double> {
  template <typename func_t>
  inline __device__ double operator() (double * address, double val, const func_t& func) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, func(val, assumed));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
  }
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)

static inline __device__ long long dummy(double val, unsigned long long int assumed) {
	return __double_as_longlong(val + __longlong_as_double(assumed));
}
// from CUDA C Programmic Guide
static inline __device__ double atomicAdd(double* address, double val)
#if defined(__clang__) && defined(__CUDA__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgcc-compat"
    __attribute__((enable_if(true, "")))
#pragma GCC diagnostic pop
#endif
{

  return AtomicFPOp<double>()(address, val, dummy);
}
#elif !defined(__CUDA_ARCH__) && (CUDA_VERSION < 8000) || defined(__HIP_PLATFORM_HCC__)
/* Note [hip-clang differences to hcc]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * The upcoming hip-clang compiler for ROCm differs from hcc in a few details.
 * It exports the __HIP__ macro, we can hence differentiate between hcc and
 * hip-clang. In the below, hcc only received support for atomicAdd with double
 * typing after work week 18312. hip-clang had support from the first version.
 * In general, the code-visible differences between hip-clang and hcc will be
 * minimal.
 */

#if defined(__HIP_PLATFORM_HCC__) && __hcc_workweek__ < 18312 && !__HIP__
  // This needs to be defined for the host side pass
  static inline  __device__  double atomicAdd(double *address, double val) { }
#endif
#endif

#define UNROLL 4
__global__ void JacobiLeftMultiplyKernel(long* dIndices, T* dResidual, T* dJ,
	T* dJtr, int numResiduals, int numDimP, int numDimJ, int numVar, int varDim, int numP, int indicesIdx, int indicesDim)
{
	int jj = blockIdx.x * blockDim.x + threadIdx.x;
	int j = jj / numDimP;
	int l = jj % numDimP;
	if (j >= numResiduals)
		return;

	//int id = dIndices[j * numVar + varDim];
	T* ptrResidual = dResidual + j * numDimJ;
	T* ptrJ = dJ + (j * indicesDim + indicesIdx) * numDimP;

	T* ptrJtr = dJtr + j * numP;

	T val = 0;
	for (int k = 0; k < numDimJ; ++k) {
		val += ptrJ[k * numResiduals * numDimP * indicesDim + l] * ptrResidual[k];
	}
	ptrJtr[l] = val;
}

__global__ void JacobiRightMultiplyKernel(
	T* dP, T* dJ, long* dIndices, T* dResidual,
	int numResiduals, int numDimJ, int numDimP, int numVar, int varDim, int indicesDim)
{
	int jj = blockIdx.x * blockDim.x + threadIdx.x;
	int j = jj / numDimJ;
	int k = jj % numDimJ;
	if (j >= numResiduals)
		return;

	for (int dim = 0; dim < indicesDim; ++dim) {
		int id = dIndices[j * indicesDim + dim];
		T* ptrP = dP + id * numDimP;

		T* ptrJ = dJ + numDimP * (
			(numResiduals * k + j) * indicesDim + dim);
		T* ptrResidual = dResidual + j * numDimJ + k;
		T val = 0;
		for (int l = 0; l < numDimP; ++l) {
			val += ptrJ[l] * ptrP[l];
		}
		*ptrResidual += val;
	}
}

__global__ void JacobiBlockJtJKernel(T* dJ, long* dIndices, T* dJtJ,
	int numDimJ, int numResiduals, int numDimP, int numVar, int varDim, int indicesDim)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= numResiduals)
		return;

	for (int dim = 0; dim < indicesDim; ++dim) {
		int id = dIndices[r * indicesDim + dim];

		T* ptrJtJ = dJtJ + id * numDimP * numDimP;
		for (int k = 0; k < numDimP; ++k) {
			for (int l = 0; l < numDimP; ++l) {
				T buf = 0;
				for (int j = 0; j < numDimJ; ++j) {
					T* ptrJ_start = dJ + j * numResiduals * numDimP * indicesDim;
					T* ptrJ = ptrJ_start + (r * indicesDim + dim) * numDimP;
					buf += ptrJ[k] * ptrJ[l];
				}
				atomicAdd(ptrJtJ + (k * numDimP + l), buf);
			}
		}
	}
}

__global__ void JacobiBlockAddDiagonalKernel(T* dJtJ, T* dD, int numV, int numP)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= numV)
		return;
	T* ptrJtJ = dJtJ + j * numP * numP;
	T* ptrD = dD + j * numP;
	for (int k = 0; k < numP; ++k) {
		ptrJtJ[k * numP + k] += ptrD[k] * ptrD[k];
	}
}

__global__ void ColumnSquareKernel(long* dIndices, T* dColumnNorm, T* dJacobian,
	int residualNum, int residualDim, int numDimP, int numDimJ, int varDim, int indicesDim)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= residualNum)
		return;

	for (int dim = 0; dim < indicesDim; ++dim) {
		int index = dIndices[j * indicesDim + dim];
		T* ptrColumnNorm = dColumnNorm + index * numDimP;
		for (int k = 0; k < numDimP; ++k) {
			T* ptrJacobian = dJacobian + (j * indicesDim + dim) * numDimP + k;
			T sum = 0;
			for (int l = 0; l < numDimJ; ++l) {
				sum += *ptrJacobian * *ptrJacobian;
				ptrJacobian += residualNum * numDimP * indicesDim;
			}
			atomicAdd(ptrColumnNorm + k, sum);
		}
	}
}

__global__ void ColumnNormInverseKernel(T* data, int num) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= num)
		return;
	data[j] = 1.0 / (1.0 + sqrt(data[j]));
}

__global__ void JacobianNormalizeKernel(
	long* dIndices, T* dColumnNorm, T* dJacobian,
	int residualNum, int residualDim, int numDimP, int numDimJ, int varDim, int indicesDim) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= residualNum)
		return;

	for (int dim = 0; dim < indicesDim; ++dim) {
		int index = dIndices[j * indicesDim + dim];
		T* ptrColumnNorm = dColumnNorm + index * numDimP;
		T* ptrJacobian = dJacobian + (j * indicesDim + dim) * numDimP;
		for (int k = 0; k < numDimJ; ++k) {
			for (int l = 0; l < numDimP; ++l) {
				ptrJacobian[l] *= ptrColumnNorm[l];
			}
			ptrJacobian += residualNum * numDimP * indicesDim;
		}
	}
}

__global__ void ListRightMultiplyKernel(T* dA, T* dX, T* dB,
	int numBlocks, int numDim)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= numBlocks)
		return;
	T* ptrA = dA + j * (numDim * numDim);
	T* ptrX = dX + j * numDim;
	T* ptrB = dB + j * numDim;
	for (int k = 0; k < numDim; ++k) {
		for (int l = 0; l < numDim; ++l) {
			ptrB[k] += ptrA[l] * ptrX[l];
		}
		ptrA += numDim;
	}
}

__global__ void SquareDotKernel(T* dJtr, T* dD, T* dP, int s)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= s)
		return;

	T buf = dD[j];
	dJtr[j] += buf * buf * dP[j];
}


void JacobiRightMultiplyGPU(T* dP, T* dJ, long* dIndices, T* dResidual,
	int numResiduals, int numDimJ, int numDimP, int numVar, int varDim, int indicesDim)
{
	JacobiRightMultiplyKernel
		<<<(numResiduals*numDimJ+256-1)/256, 256>>>(
		dP, dJ, dIndices, dResidual, numResiduals, numDimJ,
		numDimP, numVar, varDim, indicesDim);
}

void JacobiLeftMultiplyGPU(long* dIndices, T* dResidual, T* dJ, T* dJtr,
	int numResiduals, int numDimP, int numDimJ, int numVar, int varDim, int numP,
	int indicesIdx, int indicesDim)
{
	JacobiLeftMultiplyKernel
		<<<(numResiduals*numDimP+255)/256, 256>>>(
		dIndices, dResidual, dJ, dJtr,
		numResiduals, numDimP, numDimJ, numVar, varDim, numP, indicesIdx, indicesDim);
}

void JacobiBlockJtJGPU(T* dJ, long* dIndices, T* dJtJ,
	int numDimJ, int numResiduals, int numDimP, int numVar, int varDim, int indicesDim)
{
	JacobiBlockJtJKernel<<<(numResiduals+255)/256, 256>>>(
		dJ, dIndices, dJtJ, numDimJ, numResiduals, numDimP, numVar, varDim, indicesDim);
}

void JacobiBlockAddDiagonalGPU(T* dJtJ, T* dD, int numV, int numP)
{
	JacobiBlockAddDiagonalKernel<<<(numV+255)/256, 256>>>(
		dJtJ, dD, numV, numP);
}


void ColumnSquareGPU(long* dIndices, T* dColumnNorm, T* dJacobian,
	int numResiduals, int residualDim, int numDimP, int numDimJ, int varDim, int indicesDim) {

	ColumnSquareKernel<<<(numResiduals+255)/256, 256>>>(
		dIndices, dColumnNorm, dJacobian,
		numResiduals, residualDim, numDimP, numDimJ, varDim, indicesDim);
}

void ColumnNormInverseGPU(T* data, int num) {
	ColumnNormInverseKernel<<<(num+255)/256, 256>>>(data, num);
}

void JacobianNormalizeGPU(long* dIndices, T* dColumnNorm, T* dJacobian,
	int residualNum, int residualDim, int numDimP, int numDimJ, int varDim, int indicesDim) {
	JacobianNormalizeKernel<<<(residualNum+255)/256, 256>>>(
		dIndices, dColumnNorm, dJacobian, residualNum, residualDim, numDimP,
		numDimJ, varDim, indicesDim);
}

void ListRightMultiplyGPU(T* A, T* X, T* B, int numBlocks, int numDim) {
	ListRightMultiplyKernel<<<(numBlocks+255)/256, 256>>>(A, X, B,
		numBlocks, numDim);	
}

void SquareDotGPU(T* dJtr, T* dD, T* dP, int s)
{
	SquareDotKernel<<<(s+255)/256, 256>>>(dJtr, dD, dP, s);
}