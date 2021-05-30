#include "io.h"

#include "torch_util.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>


std::vector<torch::Tensor> LoadBALFromFile(const char* filename,
	int featureDim, int cameraDim, int pointDim)
{
	const int kFeatureDim = featureDim;
	const int kCameraDim = cameraDim;
	const int kPointDim = pointDim;

	int numPoints, numCameras, numObservations;

	FILE* fp = fopen(filename, "r");
	fscanf(fp, "%d %d %d", &numCameras, &numPoints, &numObservations);

	std::vector<double> features2d;
	std::vector<long> pointIndices, camIndices;
	pointIndices.resize(numObservations);
	camIndices.resize(numObservations);
	features2d.resize(numObservations * kFeatureDim);
	for (int i = 0; i < numObservations; ++i) {
		if (i % 1000 == 0) {
			printf("Load observation %d of %d...       \r", i, numObservations);
			fflush(stdout);
		}
		int camIdx, pointIdx;
		double x, y;
		fscanf(fp, "%d %d %lf %lf", &camIdx, &pointIdx, &x, &y);
		pointIndices[i] = pointIdx;
		camIndices[i] = camIdx;
		features2d[i * kFeatureDim] = x;
		features2d[i * kFeatureDim + 1] = y;
	}
	printf("\n");


	std::vector<double> cameraParameters;
	cameraParameters.resize(numCameras * kCameraDim);
	for (int i = 0; i < cameraParameters.size(); ++i)
		fscanf(fp, "%lf", &cameraParameters[i]);

	std::vector<double> points3d;
	points3d.resize(numPoints * kPointDim);
	for (int i = 0; i < points3d.size(); ++i) {
		fscanf(fp, "%lf", &points3d[i]);
	}
	fclose(fp);	

	auto floatOptions = torch::TensorOptions().dtype(torch::kFloat64);

	auto tCamera = torch::full({numCameras, kCameraDim},
		0, floatOptions);
	auto dCamera = static_cast<double*>(tCamera.storage().data());
	memcpy(dCamera, cameraParameters.data(),
		sizeof(double) * cameraParameters.size());

	auto tPoint = torch::full({numPoints, kPointDim},
		0, floatOptions);
	auto dPoint = static_cast<double*>(tPoint.storage().data());
	memcpy(dPoint, points3d.data(), sizeof(double) * points3d.size());

	auto tPtIndices = TensorFromIndices(pointIndices);
	auto tCamIndices = TensorFromIndices(camIndices);

	auto tFeat = torch::full({numObservations, kFeatureDim}, 0, floatOptions);
	auto dFeat = static_cast<double*>(tFeat.storage().data());
	memcpy(dFeat, features2d.data(), sizeof(double) * features2d.size());

	return {tPoint, tCamera, tFeat, tPtIndices, tCamIndices};
}
