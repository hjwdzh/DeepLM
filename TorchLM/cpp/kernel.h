#ifndef TORCHLM_KERNEL_H_
#define TORCHLM_KERNEL_H_

#include <vector>

#include <torch/extension.h>

void JacobiRightMultiply(
	const std::vector<torch::Tensor>& jacobians,
	const std::vector<torch::Tensor>& p,
	const std::vector<torch::Tensor>& indices,
	torch::Tensor& residual);

void JacobiLeftMultiply(
	const std::vector<torch::Tensor>& jacobians,
	const torch::Tensor& residual,
	const std::vector<torch::Tensor>& indices,
	const torch::Tensor& buffer,
	std::vector<torch::Tensor>& jtrs,
	int reinitialize);

void SquareDot(
	const std::vector<torch::Tensor>& diagonal,
	const std::vector<torch::Tensor>& p,
	std::vector<torch::Tensor>& jtrs);

void JacobiBlockJtJ(
	const std::vector<torch::Tensor>& jacobians,
	const std::vector<torch::Tensor>& diagonal,
	const std::vector<torch::Tensor>& indices,
	std::vector<torch::Tensor>& jtjs,
	int reinitialize);

void ListRightMultiply(
	const std::vector<torch::Tensor>& A,
	const std::vector<torch::Tensor>& X,
	std::vector<torch::Tensor>& B);

void JacobiColumnSquare(const std::vector<torch::Tensor>& indices,
	const std::vector<torch::Tensor>& jacobians,
	std::vector<torch::Tensor>& jacobianScale,
	int reinitialize);

void ColumnInverseSquare(std::vector<torch::Tensor>& jacobianScale);

void JacobiNormalize(const std::vector<torch::Tensor>& indices,
	const std::vector<torch::Tensor>& jacobianScale,
	std::vector<torch::Tensor>& jacobians);

void JacobiRightMultiplyCuda(
	const std::vector<torch::Tensor>& jacobians,
	const std::vector<torch::Tensor>& p,
	const std::vector<torch::Tensor>& indices,
	torch::Tensor& residual);

void JacobiLeftMultiplyCuda(
	const std::vector<torch::Tensor>& jacobians,
	const torch::Tensor& residual,
	const std::vector<torch::Tensor>& indices,
	const torch::Tensor& buffer,
	std::vector<torch::Tensor>& jtrs,
	int reinitialize);

void JacobiColumnSquareCuda(const std::vector<torch::Tensor>& indices,
	const std::vector<torch::Tensor>& jacobians,
	std::vector<torch::Tensor>& jacobianScale,
	int reinitialize);

void ColumnInverseSquareCuda(std::vector<torch::Tensor>& jacobianScale);

void JacobiNormalizeCuda(const std::vector<torch::Tensor>& indices,
	const std::vector<torch::Tensor>& jacobianScale,
	std::vector<torch::Tensor>& jacobians);

void JacobiBlockJtJCuda(
	const std::vector<torch::Tensor>& jacobians,
	const std::vector<torch::Tensor>& diagonal,
	const std::vector<torch::Tensor>& indices,
	std::vector<torch::Tensor>& jtjs,
	int reinitialize);

void ListRightMultiplyCuda(
	const std::vector<torch::Tensor>& A,
	const std::vector<torch::Tensor>& X,
	std::vector<torch::Tensor>& B);
//std::vector<torch::Tensor> AnalyzeReducedIndices(const torch::Tensor& I);

#endif