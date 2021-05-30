#include "torch_util.h"

torch::Tensor TensorFromIndices(const std::vector<long>& indices) {
	auto intOptions = torch::TensorOptions().dtype(torch::kInt64);
	auto res = torch::full({(long)indices.size()}, 0, intOptions);
	auto data = static_cast<long*>(res.storage().data());
	memcpy(data, indices.data(), sizeof(long) * indices.size());
	return res;
}
