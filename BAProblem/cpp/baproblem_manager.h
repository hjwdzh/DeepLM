#ifndef BAPROBLEM_BAPROBLEM_MANAGER_H_
#define BAPROBLEM_BAPROBLEM_MANAGER_H_

#include <vector>

#include <torch/extension.h>

torch::Tensor SortIndices(torch::Tensor& pointIdx, torch::Tensor& cameraIdx,
	torch::Tensor& features, long numPoints);

std::vector<std::vector<torch::Tensor> > PrepareSeparator(
	const torch::Tensor& pointIdx,
	const torch::Tensor& cameraIdx,
	const torch::Tensor& startIdx,
	const torch::Tensor& cameraLabels,
	torch::Tensor& separatorMask,
	long maxIndices);

std::vector<std::vector<torch::Tensor> > PrepareBlocks(
	const torch::Tensor& pointIdx,
	const torch::Tensor& cameraIdx,
	const torch::Tensor& startIdx,
	const torch::Tensor& cameraLabels,
	const torch::Tensor& separatorMask,
	long numBlocks);

#endif