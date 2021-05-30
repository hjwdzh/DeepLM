#include "kernel.h"

#include "kernel_impl.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

void JacobiRightMultiply(
	const std::vector<torch::Tensor>& jacobians,
	const std::vector<torch::Tensor>& p,
	const std::vector<torch::Tensor>& indices,
	torch::Tensor& residual)
{
	if (indices[0].device().type() == torch::kCUDA) {
		JacobiRightMultiplyCuda(jacobians, p, indices, residual);
		return;
	}

	int numResiduals = jacobians[0].size(1);
	int numVar = jacobians.size();
	int numDimJ = jacobians[0].size(0);

	auto dResidual = static_cast<T*>(residual.storage().data());
	memset(dResidual, 0, sizeof(T) * numResiduals * numDimJ);

	for (int i = 0; i < numVar; ++i) {
		int indicesDim = indices[i].size(1);
		long* dIndices = static_cast<long*>(indices[i].storage().data());
		auto& J = jacobians[i];
		auto& P = p[i];
		int numDimP = P.size(1);
		auto dP = static_cast<T*>(P.storage().data());
		auto dJ = static_cast<T*>(J.storage().data());

#pragma omp parallel for
		for (int j = 0; j < numResiduals; ++j) {
			for (int dim = 0; dim < indicesDim; ++dim) {
				int id = dIndices[j * indicesDim + dim];
				T* ptrP = dP + id * numDimP;

				for (int k = 0; k < numDimJ; ++k) {
					T* ptrJ = dJ + numDimP * (
						(numResiduals * k + j) * indicesDim + dim);
					T* ptrResidual = dResidual + j * numDimJ + k;
					for (int l = 0; l < numDimP; ++l) {
						*ptrResidual += ptrJ[l] * ptrP[l];
					}
				}
			}
		}
	}
}

void JacobiLeftMultiply(
	const std::vector<torch::Tensor>& jacobians,
	const torch::Tensor& residual,
	const std::vector<torch::Tensor>& indices,
	const torch::Tensor& buffer,
	std::vector<torch::Tensor>& jtrs,
	int reinitialize)
{
	if (residual.device().type() == torch::kCUDA) {
		JacobiLeftMultiplyCuda(jacobians, residual,
			indices, buffer, jtrs, reinitialize);
		return;
	}
	int numResiduals = jacobians[0].size(1);
	int numVar = jacobians.size();
	int numDimJ = jacobians[0].size(0);

	auto dResidual = static_cast<T*>(residual.storage().data());
#pragma omp parallel for
	for (int i = 0; i < numVar; ++i) {
		int indicesDim = indices[i].size(1);
		auto& jtr = jtrs[i];
		auto& J = jacobians[i];
		long* dIndices = static_cast<long*>(indices[i].storage().data());

		int numDimV = jtr.size(0);
		int numDimP = jtr.size(1);

		auto dJ = static_cast<T*>(J.storage().data());
		auto dJtr = static_cast<T*>(jtr.storage().data());

		//auto dEndIdx = static_cast<long*>(endIdx[i].storage().data());
		//auto dIndicesIdx = static_cast<long*>(
		//	indicesIdx[i].storage().data());

		if (reinitialize)
			memset(dJtr, 0, sizeof(T) * numDimV * numDimP);

//#pragma omp parallel for
		for (int j = 0; j < numResiduals; ++j) {
			for (int dim = 0; dim < indicesDim; ++dim) {
				int id = dIndices[j * indicesDim + dim];
				T* ptrResidual = dResidual + j * numDimJ;
				T* ptrJ = dJ + (j * indicesDim + dim) * numDimP;
				T* ptrJtr = dJtr + id * numDimP;
				for (int k = 0; k < numDimJ; ++k) {
					for (int l = 0; l < numDimP; ++l) {
						T* ptr = ptrJtr + l;
						T val = ptrJ[l] * ptrResidual[k];
//#pragma omp atomic
						*ptr += val;
					}
					ptrJ += numResiduals * numDimP * indicesDim;
				}
			}
		}
	}	
}

void SquareDot(
	const std::vector<torch::Tensor>& diagonal,
	const std::vector<torch::Tensor>& p,
	std::vector<torch::Tensor>& jtrs)
{
	int numVar = p.size();

#ifdef WITH_CUDA
	if (p[0].device().type() == torch::kCUDA) {
		for (int i = 0; i < numVar; ++i) {
			auto& jtr = jtrs[i];
			auto dJtr = jtr.data_ptr<T>();
			auto dD = diagonal[i].data_ptr<T>();
			auto dP = p[i].data_ptr<T>();

			int numDimV = jtr.size(0);
			int numDimP = jtr.size(1);

			int s = numDimV * numDimP;
			SquareDotGPU(dJtr, dD, dP, s);
		}
		return;
	}
#endif
	for (int i = 0; i < numVar; ++i) {
		auto& jtr = jtrs[i];
		auto dJtr = static_cast<T*>(jtr.storage().data());
		auto dD = static_cast<T*>(diagonal[i].storage().data());
		auto dP = static_cast<T*>(p[i].storage().data());

		int numDimV = jtr.size(0);
		int numDimP = jtr.size(1);

		int s = numDimV * numDimP;
#pragma omp parallel for
		for (int j = 0; j < s; ++j) {
			dJtr[j] += dD[j] * dD[j] * dP[j];
		}
	}

}

void JacobiBlockJtJ(
	const std::vector<torch::Tensor>& jacobians,
	const std::vector<torch::Tensor>& diagonal,
	const std::vector<torch::Tensor>& indices,
	std::vector<torch::Tensor>& jtjs,
	int reinitialize)
{
	if (indices[0].device().type() == torch::kCUDA) {
		JacobiBlockJtJCuda(jacobians, diagonal, indices, jtjs, reinitialize);
		return;
	}
	int numVar = jtjs.size();

#pragma omp parallel for
	for (int i = 0; i < numVar; ++i) {
		int indicesDim = indices[i].size(1);
		long* dIndices = static_cast<long*>(indices[i].storage().data());
		auto& J = jacobians[i];
		//auto& D = diagonal[i];
		auto dJ = static_cast<T*>(J.storage().data());
		auto dD = static_cast<T*>(diagonal[i].storage().data());

		auto dJtJ = static_cast<T*>(jtjs[i].storage().data());

		int numDimJ = J.size(0);
		int numResiduals = J.size(1);
		int numDimP = J.size(3);
		int numDimV = jtjs[i].size(0);

		if (reinitialize)
			memset(dJtJ, 0, sizeof(T) * numDimV * numDimP * numDimP);

		for (int j = 0; j < numDimJ; ++j) {
			T* ptrJ_start = dJ + j * numResiduals * numDimP * indicesDim;
			for (int r = 0; r < numResiduals; ++r) {
				for (int dim = 0; dim < indicesDim; ++dim) {
					int id = dIndices[r * indicesDim + dim];
					T* ptrJ = ptrJ_start + (r * indicesDim + dim) * numDimP;
					T* ptrJtJ = dJtJ + id * numDimP * numDimP;
					for (int k = 0; k < numDimP; ++k) {
						for (int l = 0; l < numDimP; ++l) {
							ptrJtJ[k * numDimP + l] += ptrJ[k] * ptrJ[l];
						}
					}
				}
			}
		}

		for (int j = 0; j < numDimV; ++j) {
			T* ptrJtJ = dJtJ + j * numDimP * numDimP;
			T* ptrD = dD + j * numDimP;
			for (int k = 0; k < numDimP; ++k) {
				ptrJtJ[k * numDimP + k] += ptrD[k] * ptrD[k];
			}
		}
	}
}

void ListRightMultiply(
	const std::vector<torch::Tensor>& A,
	const std::vector<torch::Tensor>& X,
	std::vector<torch::Tensor>& B) {
	if (A.size() > 0 && A[0].device().type() == torch::kCUDA) {
		ListRightMultiplyCuda(A, X, B);
		return;
	}

	int numVar = A.size();
	for (int i = 0; i < numVar; ++i) {
		int numBlocks = A[i].size(0);
		int numDim = A[i].size(1);

		auto dA = static_cast<T*>(A[i].storage().data());
		auto dX = static_cast<T*>(X[i].storage().data());
		auto dB = static_cast<T*>(B[i].storage().data());

		memset(dB, 0, sizeof(T) * numDim * numBlocks);
#pragma omp parallel for
		for (int j = 0; j < numBlocks; ++j) {
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
	}
}

void JacobiColumnSquare(const std::vector<torch::Tensor>& indices,
	const std::vector<torch::Tensor>& jacobians,
	std::vector<torch::Tensor>& jacobianScale,
	int reinitialize)
{
	if (indices[0].device().type() == torch::kCUDA) {
		JacobiColumnSquareCuda(indices, jacobians, jacobianScale, reinitialize);
		return;
	}
	int residualNum = indices[0].size(0);
	int residualDim = indices.size();

#pragma omp parallel for
	for (int i = 0; i < jacobians.size(); ++i) {
		//printf("i %d\n", i);
		int indicesDim = indices[i].size(1);
		auto dIndices = static_cast<long*>(indices[i].storage().data());
		auto dColumnNorm = static_cast<T*>(jacobianScale[i].storage().data());
		auto dJacobian = static_cast<T*>(jacobians[i].storage().data());

		int numDimJ = jacobians[i].size(0);
		int numDimP = jacobianScale[i].size(1);
		int numDimV = jacobianScale[i].size(0);
		if (reinitialize)
			memset(dColumnNorm, 0, sizeof(T) * numDimP * numDimV);
		for (int j = 0; j < residualNum; ++j) {
			//printf("j %d\n", j);
			for (int dim = 0; dim < indicesDim; ++dim) {
				int index = dIndices[j * indicesDim + dim];
				T* ptrColumnNorm = dColumnNorm + index * numDimP;
				for (int k = 0; k < numDimP; ++k) {
					//printf("k %d\n", j);
					T* ptrJacobian = dJacobian
						+ (j * indicesDim + dim) * numDimP + k;
					T sum = 0;
					for (int l = 0; l < numDimJ; ++l) {
						sum += *ptrJacobian * *ptrJacobian;
						ptrJacobian += residualNum * numDimP * indicesDim;
					}
					// this is atomic
					ptrColumnNorm[k] += sum;
				}
			}
		}
	}
}

void ColumnInverseSquare(std::vector<torch::Tensor>& jacobianScale)
{
	if (jacobianScale[0].device().type() == torch::kCUDA) {
		ColumnInverseSquareCuda(jacobianScale);
		return;
	}
	for (int i = 0; i < jacobianScale.size(); ++i) {
		auto dColumnNorm = static_cast<T*>(jacobianScale[i].storage().data());
		int numDimV = jacobianScale[i].size(0);
		int numDimP = jacobianScale[i].size(1);
		int num = numDimV * numDimP;
#pragma omp parallel for
		for (int j = 0; j < num; ++j) {
			dColumnNorm[j] = 1.0 / (1.0 + sqrt(dColumnNorm[j]));
		}
	}
}

void JacobiNormalize(const std::vector<torch::Tensor>& indices,
	const std::vector<torch::Tensor>& jacobianScale,
	std::vector<torch::Tensor>& jacobians)
{
	if (indices[0].device().type() == torch::kCUDA) {
		JacobiNormalizeCuda(indices, jacobianScale, jacobians);
		return;
	}

	//JacobiColumnSquare(indices, jacobians, jacobianScale);

	int residualNum = indices[0].size(0);
	int residualDim = indices.size();

	for (int i = 0; i < jacobianScale.size(); ++i) {
		int indicesDim = indices[i].size(1);
		auto dIndices = static_cast<long*>(indices[i].storage().data());
		auto dColumnNorm = static_cast<T*>(jacobianScale[i].storage().data());
		auto dJacobian = static_cast<T*>(jacobians[i].storage().data());
		int numDimJ = jacobians[i].size(0);
		int numDimV = jacobianScale[i].size(0);
		int numDimP = jacobianScale[i].size(1);
		int num = numDimV * numDimP;
#pragma omp parallel for
		for (int j = 0; j < residualNum; ++j) {
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
	}
}

#ifdef WITH_CUDA
void JacobiRightMultiplyCuda(
	const std::vector<torch::Tensor>& jacobians,
	const std::vector<torch::Tensor>& p,
	const std::vector<torch::Tensor>& indices,
	torch::Tensor& residual)
{
	int numResiduals = jacobians[0].size(1);
	int numVar = jacobians.size();
	int numDimJ = jacobians[0].size(0);

	auto dResidual = residual.data_ptr<T>();
	cudaMemset(dResidual, 0, sizeof(T) * numResiduals * numDimJ);

	for (int i = 0; i < numVar; ++i) {
		int indicesDim = indices[i].size(1);
		long* dIndices = indices[i].data_ptr<long>();
		auto& J = jacobians[i];
		auto& P = p[i];
		int numDimP = P.size(1);
		auto dP = P.data_ptr<T>();
		auto dJ = J.data_ptr<T>();

		//cudaDeviceSynchronize();
		//auto t1 = std::chrono::high_resolution_clock::now();

		JacobiRightMultiplyGPU(dP, dJ, dIndices, dResidual,
			numResiduals, numDimJ, numDimP, numVar, i, indicesDim);
		//cudaDeviceSynchronize();
		//auto t2 = std::chrono::high_resolution_clock::now();
		//auto d1 = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
		//printf("%d %d\n", i, d1);
	}
	//exit(0);
}

void JacobiLeftMultiplyCuda(
	const std::vector<torch::Tensor>& jacobians,
	const torch::Tensor& residual,
	const std::vector<torch::Tensor>& indices,
	const torch::Tensor& buffer,
	std::vector<torch::Tensor>& jtrs,
	int reinitialize)
{
	int numResiduals = jacobians[0].size(1);
	int numVar = jacobians.size();
	int numDimJ = jacobians[0].size(0);

	auto dResidual = residual.data_ptr<T>();

	for (int i = 0; i < numVar; ++i) {
		long* dIndices = (long*)indices[i].data_ptr<long>();
		auto& jtr = jtrs[i];
		auto& J = jacobians[i];

		int numDimV = jtr.size(0);
		int numDimP = jtr.size(1);
		int numP = buffer.size(1);

		T* dJ = J.data_ptr<T>();//static_cast<T*>(J.storage().data());

		//auto dEndIdx = endIdx[i].data_ptr<long>();
		//auto dIndicesIdx = indicesIdx[i].data_ptr<long>();
		if (reinitialize)
			jtr.zero_();

		for (int j = 0; j < indices[i].size(1); ++j) {		
			buffer.zero_();
			T* dJtr = buffer.data_ptr<T>();
			//cudaMemset(dJtr, 0, sizeof(T) * numResiduals * numP);

			JacobiLeftMultiplyGPU(
				dIndices, dResidual, dJ, dJtr,
				numResiduals, numDimP, numDimJ, numVar, i, numP, j,
				indices[i].size(1));

			jtr.index_add_(0, indices[i].narrow(1,j,1).view(-1),
				buffer.narrow(1,0,numDimP));
		}
		//T* dJtr1 = jtr.data_ptr<T>();
		//cudaMalloc(&dJtr1, sizeof(double) * numDimP * numResiduals);

		cudaDeviceSynchronize();
	}
}

void JacobiBlockJtJCuda(
	const std::vector<torch::Tensor>& jacobians,
	const std::vector<torch::Tensor>& diagonal,
	const std::vector<torch::Tensor>& indices,
	std::vector<torch::Tensor>& jtjs,
	int reinitialize) {

	int numVar = jtjs.size();

	for (int i = 0; i < numVar; ++i) {
		int indicesDim = indices[i].size(1);
		long* dIndices = indices[i].data_ptr<long>();
		auto& J = jacobians[i];
		//auto& D = diagonal[i];
		auto dJ = J.data_ptr<T>();
		auto dD = diagonal[i].data_ptr<T>();

		auto dJtJ = jtjs[i].data_ptr<T>();

		int numDimJ = J.size(0);
		int numResiduals = J.size(1);
		int numDimP = J.size(3);
		int numDimV = jtjs[i].size(0);

		if (reinitialize)
			cudaMemset(dJtJ, 0, sizeof(T) * numDimV * numDimP * numDimP);

		JacobiBlockJtJGPU(dJ, dIndices, dJtJ, numDimJ, numResiduals,
			numDimP, numVar, i, indicesDim);

		JacobiBlockAddDiagonalGPU(dJtJ, dD, numDimV, numDimP);
	}
}

void JacobiColumnSquareCuda(const std::vector<torch::Tensor>& indices,
	const std::vector<torch::Tensor>& jacobians,
	std::vector<torch::Tensor>& jacobianScale,
	int reinitialize)
{
	int residualNum = indices[0].size(0);
	int residualDim = indices.size();

	for (int i = 0; i < jacobians.size(); ++i) {
		int indicesDim = indices[i].size(1);
		auto dIndices = indices[i].data_ptr<long>();
		auto dColumnNorm = jacobianScale[i].data_ptr<T>();
		auto dJacobian = jacobians[i].data_ptr<T>();

		int numDimJ = jacobians[i].size(0);
		int numDimP = jacobianScale[i].size(1);
		int numDimV = jacobianScale[i].size(0);
		if (reinitialize)
			cudaMemset(dColumnNorm, 0, sizeof(T) * numDimP * numDimV);

		ColumnSquareGPU(dIndices, dColumnNorm, dJacobian, residualNum,
			residualDim, numDimP, numDimJ, i, indicesDim);
	}	
}

void ColumnInverseSquareCuda(std::vector<torch::Tensor>& jacobianScale) {
	for (int i = 0; i < jacobianScale.size(); ++i) {
		auto dColumnNorm = jacobianScale[i].data_ptr<T>();
		int numDimV = jacobianScale[i].size(0);
		int numDimP = jacobianScale[i].size(1);
		int num = numDimV * numDimP;
		ColumnNormInverseGPU(dColumnNorm, num);
	}	
}

void JacobiNormalizeCuda(const std::vector<torch::Tensor>& indices,
	const std::vector<torch::Tensor>& jacobianScale,
	std::vector<torch::Tensor>& jacobians)
{
	//JacobiColumnSquareCuda(indices, jacobians, jacobianScale);
	int residualNum = indices[0].size(0);
	int residualDim = indices.size();

	for (int i = 0; i < jacobianScale.size(); ++i) {
		int indicesDim = indices[i].size(1);
		auto dIndices = indices[i].data_ptr<long>();
		auto dColumnNorm = jacobianScale[i].data_ptr<T>();
		auto dJacobian = jacobians[i].data_ptr<T>();
		int numDimJ = jacobians[i].size(0);
		int numDimV = jacobianScale[i].size(0);
		int numDimP = jacobianScale[i].size(1);
		int num = numDimV * numDimP;

		JacobianNormalizeGPU(dIndices, dColumnNorm, dJacobian, residualNum,
			residualDim, numDimP, numDimJ, i, indicesDim);
	}
}

void ListRightMultiplyCuda(
	const std::vector<torch::Tensor>& A,
	const std::vector<torch::Tensor>& X,
	std::vector<torch::Tensor>& B) {
	int numVar = A.size();
	for (int i = 0; i < numVar; ++i) {
		int numBlocks = A[i].size(0);
		int numDim = A[i].size(1);

		auto dA = A[i].data_ptr<T>();
		auto dX = X[i].data_ptr<T>();
		auto dB = B[i].data_ptr<T>();


		cudaMemset(dB, 0, sizeof(T) * numDim * numBlocks);

		ListRightMultiplyGPU(dA, dX, dB, numBlocks, numDim);
	}
}

#else
void JacobiRightMultiplyCuda(
	const std::vector<torch::Tensor>& jacobians,
	const std::vector<torch::Tensor>& p,
	const std::vector<torch::Tensor>& indices,
	torch::Tensor& residual) {
	printf("No cuda implementation!\n");
	exit(0);
}

void JacobiLeftMultiplyCuda(
	const std::vector<torch::Tensor>& jacobians,
	const torch::Tensor& residual,
	const std::vector<torch::Tensor>& indices,
	const torch::Tensor& buffer,
	std::vector<torch::Tensor>& jtrs) {
	printf("No cuda implementation!\n");
	exit(0);
}

void JacobiColumnSquareCuda(const std::vector<torch::Tensor>& indices,
	const std::vector<torch::Tensor>& jacobians,
	std::vector<torch::Tensor>& jacobianScale) {
	printf("No cuda implementation!\n");
	exit(0);
}

void JacobiNormalizeCuda(const std::vector<torch::Tensor>& indices,
	std::vector<torch::Tensor>& jacobians,
	std::vector<torch::Tensor>& jacobianScale) {
	printf("No cuda implementation!\n");
	exit(0);
}

void JacobiBlockJtJCuda(
	const std::vector<torch::Tensor>& jacobians,
	const std::vector<torch::Tensor>& diagonal,
	const std::vector<torch::Tensor>& indices,
	std::vector<torch::Tensor>& jtjs) {
	printf("No cuda implementation!\n");
	exit(0);
}

void ListRightMultiplyCuda(
	const std::vector<torch::Tensor>& A,
	const std::vector<torch::Tensor>& X,
	std::vector<torch::Tensor>& B) {
	printf("No cuda implementation!\n");
	exit(0);
}
#endif













/*
std::vector<torch::Tensor> AnalyzeReducedIndices(const torch::Tensor& I)
{
	struct Key
	{
		Key(){}
		Key(int x, int y, int z, int id)
		: x(x), y(y), z(z), id(id)
		{}
		int x, y, z, id;
		bool operator<(const Key& n) const {
			return x < n.x ||
				x == n.x && y < n.y ||
				x == n.x && y == n.y && z < n.z;
		}
		bool operator==(const Key& n) const {
			return x == n.x && y == n.y && z == n.z;
		}
	};

	int num = I.size(0);
	const long* data = static_cast<const long*>(I.storage().data());	
	std::vector<Key> s(num);
	for (int i = 0; i < num; ++i) {
		s[i] = Key(data[i * 3], data[i * 3 + 1], data[i * 3 + 2], i);
	}
	std::sort(s.begin(), s.end());

	auto intOptions = torch::TensorOptions().dtype(torch::kInt64);

	auto compactIndices = torch::full({num}, 0, intOptions);
	auto dCompactIndices = static_cast<long*>(compactIndices.storage().data());

	int c = 0;	
	for (int i = 0; i < num; ++i) {
		dCompactIndices[s[i].id] = c;
		if (i == num - 1 || !(s[i] == s[i + 1])) {
			c += 1;
		}
	}

	auto reducedIndex = torch::full({c, 3}, 0, intOptions);
	auto dReducedIndex = static_cast<long*>(reducedIndex.storage().data());
	c = 0;	
	for (int i = 0; i < num; ++i) {
		if (i == num - 1 || !(s[i] == s[i + 1])) {
			long* offset = dReducedIndex + c * 3;
			const long* orig = data + s[i].id * 3;
			offset[0] = orig[0];
			offset[1] = orig[1];
			offset[2] = orig[2];
			c += 1;
		}
	}

	return {compactIndices, reducedIndex};
}
*/