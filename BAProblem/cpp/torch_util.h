#ifndef BAPROBLEM_TORCH_UTIL_H_
#define BAPROBLEM_TORCH_UTIL_H_

#include <vector>

#include <torch/extension.h>

torch::Tensor TensorFromIndices(const std::vector<long>& indices);

#endif