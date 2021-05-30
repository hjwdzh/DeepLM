#include "baproblem_manager.h"

#include <omp.h>
#include <vector>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "torch_util.h"

typedef double T;

torch::Tensor SortIndices(torch::Tensor& pointIdx, torch::Tensor& cameraIdx,
	torch::Tensor& features, long numPoints)
{
	long numIndices = pointIdx.size(0);
	std::vector<std::pair<std::pair<long, long>, long> > results(numIndices);
	long* ptIdx = static_cast<long*>(pointIdx.storage().data());
	long* camIdx = static_cast<long*>(cameraIdx.storage().data());
	for (long i = 0; i < numIndices; ++i) {
		results[i] = std::make_pair(std::make_pair(ptIdx[i], camIdx[i]), i);
	}
	std::sort(results.begin(), results.end());

	auto intOptions = torch::TensorOptions().dtype(torch::kInt64);
	auto startIdx = torch::full({numPoints + 1}, 0, intOptions);
	auto dStartIdx = static_cast<long*>(startIdx.storage().data());
	long currentP = 0;

	torch::Tensor featuresTemp = features.clone();
	auto dFeatures = static_cast<double*>(features.storage().data());
	auto dFeaturesTemp = static_cast<double*>(featuresTemp.storage().data());
	long featDim = features.size(1);
	for (long i = 0; i < numIndices; ++i) {
		ptIdx[i] = results[i].first.first;
		camIdx[i] = results[i].first.second;
		while (ptIdx[i] > currentP) {
			currentP += 1;
			dStartIdx[currentP] = i;
		}

		double* fData = dFeatures + i * featDim;
		double* fDataTemp = dFeaturesTemp + results[i].second * featDim;
		memcpy(fData, fDataTemp, sizeof(double) * featDim);
	}

	while (currentP < numPoints) {
		currentP += 1;
		dStartIdx[currentP] = numIndices;
	}

	return startIdx;
}

std::vector<std::vector<torch::Tensor> > PrepareSeparator(
	const torch::Tensor& pointIdx,
	const torch::Tensor& cameraIdx,
	const torch::Tensor& startIdx,
	const torch::Tensor& cameraLabels,
	torch::Tensor& separatorMask,
	long maxIndices)
{
	const long* dPtIdx = static_cast<const long*>(pointIdx.storage().data());
	const long* dCamIdx = static_cast<const long*>(cameraIdx.storage().data());
	const long* dStartIdx = static_cast<const long*>(startIdx.storage().data());
	const long* label = static_cast<const long*>(cameraLabels.storage().data());
	long* mask = static_cast<long*>(separatorMask.storage().data());

	long numIndices = pointIdx.size(0);
	long numPoints = startIdx.size(0) - 1;
	
	std::vector<std::vector<long> > originPtIndices;
	std::vector<std::vector<long> > originCamIndices;
	std::vector<std::vector<long> > compactPtIndices;

	originPtIndices.reserve(numIndices / maxIndices);
	originCamIndices.reserve(numIndices / maxIndices);
	compactPtIndices.reserve(numIndices / maxIndices);

	long currentNumIndices = maxIndices;
	for (long i = 0; i < numPoints; ++i) {
		long front = dStartIdx[i];
		long rear = dStartIdx[i + 1];
		std::unordered_set<long> blockIds;
		for (long j = front; j < rear; ++j) {
			long l = label[dCamIdx[j]];
			blockIds.insert(l);
			if (blockIds.size() == 2)
				break;
		}
		if (blockIds.size() > 1) {
			if (currentNumIndices >= maxIndices) {
				originPtIndices.push_back(std::vector<long>());
				originCamIndices.push_back(std::vector<long>());
				compactPtIndices.push_back(std::vector<long>());
				currentNumIndices = 0;
			}
			long id = originPtIndices.back().size();
			for (long j = front; j < rear; ++j) {
				originCamIndices.back().push_back(j);
				compactPtIndices.back().push_back(id);
			}			
			originPtIndices.back().push_back(i);
			currentNumIndices += (rear - front);
			mask[i] = 1;
		}
		else {
			mask[i] = 0;
		}
	}

	std::vector<torch::Tensor> tOriginPtIndices(originPtIndices.size());
	std::vector<torch::Tensor> tOriginCamIndices(originCamIndices.size());
	std::vector<torch::Tensor> tCompactPtIndices(compactPtIndices.size());

	auto intOptions = torch::TensorOptions().dtype(torch::kInt64);
	for (long i = 0; i < originPtIndices.size(); ++i) {
		tOriginPtIndices[i] = TensorFromIndices(originPtIndices[i]);
		tOriginCamIndices[i] = TensorFromIndices(originCamIndices[i]);
		tCompactPtIndices[i] = TensorFromIndices(compactPtIndices[i]);
	}

	return {tOriginPtIndices, tOriginCamIndices, tCompactPtIndices};
}

std::vector<std::vector<torch::Tensor> > PrepareBlocks(
	const torch::Tensor& pointIdx,
	const torch::Tensor& cameraIdx,
	const torch::Tensor& startIdx,
	const torch::Tensor& cameraLabels,
	const torch::Tensor& separatorMask,
	long numBlocks)
{
	const long* dPtIdx = static_cast<const long*>(pointIdx.storage().data());
	const long* dCamIdx = static_cast<const long*>(cameraIdx.storage().data());
	const long* dStartIdx = static_cast<const long*>(startIdx.storage().data());
	const long* label = static_cast<const long*>(cameraLabels.storage().data());
	const long* mask = static_cast<const long*>(separatorMask.storage().data());

	long numIndices = pointIdx.size(0);
	long numPoints = startIdx.size(0) - 1;
	long numCameras = cameraLabels.size(0);

	std::vector<std::vector<long> > originPtIdx(numBlocks);
	std::vector<std::vector<long> > originCamIdx(numBlocks);
	std::vector<std::vector<long> > originFeatIdx(numBlocks);
	std::vector<std::vector<long> > originPtSepIdx(numBlocks);
	std::vector<std::vector<long> > compactPtIdx(numBlocks);
	std::vector<std::vector<long> > compactCamIdx(numBlocks);
	std::vector<std::vector<long> > compactCamSepIdx(numBlocks);

	std::unordered_map<long, long> originToCompactCam, originToCompactPt;
	for (long i = 0; i < numCameras; ++i) {
		long l = label[i];
		originToCompactCam[i] = originCamIdx[l].size();
		originCamIdx[l].push_back(i);
	}
	for (long i = 0; i < numPoints; ++i) {
		long l = 0;
		long s = dStartIdx[i];
		long t = dStartIdx[i + 1];
		if (s < t)
			l = label[dCamIdx[s]];
		originToCompactPt[i] = originPtIdx[l].size();
		originPtIdx[l].push_back(i);
	}
	for (long i = 0; i < numIndices; ++i) {
		long pt = dPtIdx[i];
		long cam = dCamIdx[i];
		long l = label[cam];
		if (mask[pt] > 0) {
			originPtSepIdx[l].push_back(i);
			compactCamSepIdx[l].push_back(originToCompactCam[cam]);
		} else {
			originFeatIdx[l].push_back(i);
			compactPtIdx[l].push_back(originToCompactPt[pt]);
			compactCamIdx[l].push_back(originToCompactCam[cam]);
		}
	}

	std::vector<torch::Tensor> tOriginPtIdx(numBlocks);
	std::vector<torch::Tensor> tOriginCamIdx(numBlocks);
	std::vector<torch::Tensor> tOriginFeatIdx(numBlocks);
	std::vector<torch::Tensor> tOriginPtSepIdx(numBlocks);
	std::vector<torch::Tensor> tCompactPtIdx(numBlocks);
	std::vector<torch::Tensor> tCompactCamIdx(numBlocks);
	std::vector<torch::Tensor> tCompactCamSepIdx(numBlocks);

	for (int i = 0; i < numBlocks; ++i) {
		tOriginPtIdx[i] = TensorFromIndices(originPtIdx[i]);
		tOriginCamIdx[i] = TensorFromIndices(originCamIdx[i]);
		tOriginFeatIdx[i] = TensorFromIndices(originFeatIdx[i]);
		tOriginPtSepIdx[i] = TensorFromIndices(originPtSepIdx[i]);
		tCompactPtIdx[i] = TensorFromIndices(compactPtIdx[i]);
		tCompactCamIdx[i] = TensorFromIndices(compactCamIdx[i]);
		tCompactCamSepIdx[i] = TensorFromIndices(compactCamSepIdx[i]);
	}
	return {tOriginPtIdx, tOriginCamIdx, tOriginFeatIdx, tOriginPtSepIdx,
		tCompactPtIdx, tCompactCamIdx, tCompactCamSepIdx};
}