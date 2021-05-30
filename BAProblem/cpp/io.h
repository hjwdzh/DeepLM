#ifndef BAPROBLEM_IO_H_
#define BAPROBLEM_IO_H_

#include <vector>

#include <torch/extension.h>

std::vector<torch::Tensor> LoadBALFromFile(const char* filename,
	int featureDim, int cameraDim, int pointDim);

std::vector<torch::Tensor> LoadRiemannFromFile(const char* camFile,
	const char* txtFile, const char* ptbFile);

std::vector<torch::Tensor> LoadRiemannLargeFromFile(
	const char* eopFile, const char* ptbFile);

void SavePts(const char* filename, const torch::Tensor& tensor);

#endif