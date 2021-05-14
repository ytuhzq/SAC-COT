#include"CApp.h"
#include"graph.h"
#include<fstream>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include<pcl/io/pcd_io.h>

using namespace sac_cot;
using namespace std;

//read .xyz data
int CApp::XYZ_Read(const char* Filename)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	int i, nXYZ_nums;
	std::vector<Vertex> vXYZ;
	FILE* fp = fopen(Filename, "r");
	if (fp == NULL)
	{
		printf("File can't open!\n");
		std::cout << Filename << std::endl;
		system("pause");
		exit(0);
	}
	fscanf(fp, "%d\n", &nXYZ_nums);
	vXYZ.resize(nXYZ_nums);
	for (i = 0; i < vXYZ.size(); i++)
	{
		fscanf(fp, "%f %f %f\n", &vXYZ[i].x, &vXYZ[i].y, &vXYZ[i].z);
	}
	fclose(fp);
	cloud->width = vXYZ.size();
	cloud->height = 1;
	cloud->is_dense = true;
	cloud->points.resize(cloud->width * cloud->height);
	for (i = 0; i < cloud->points.size(); i++)
	{
		cloud->points[i].x = vXYZ[i].x;
		cloud->points[i].y = vXYZ[i].y;
		cloud->points[i].z = vXYZ[i].z;
	}
	LoadPointCloud(cloud);
	return 0;
}

int CApp::XYZ_Save(const char* path, const char* src, const char* tar)
{
	auto transMatrix = _matrix;
	auto cloudSrc = _pointcloud[0];
	auto cloudTar = _pointcloud[1];
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTrans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloudSrc, *cloudTrans, _matrix);
	pcl::io::savePCDFile((std::string)path + (std::string)src + ".pcd", *cloudTrans, false);
	pcl::io::savePCDFile((std::string)path + (std::string)tar + ".pcd", *cloudTar, false);
}

void CApp::LoadPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	_pointcloud.push_back(cloud);
}

//uniform dowm sampling for point cloud
void CApp::downSampling(double downsamplingSize)
{
	//calculate point cloud resolution before downsampling
	cout << "Before downsampling cloud_src->size()=" << _pointcloud[0]->size() << endl;
	computeResolution();

	_doDownSampling(_pointcloud[0], downsamplingSize);
	_doDownSampling(_pointcloud[1], downsamplingSize);
	cout << "After downsampling cloud_src->size()=" << _pointcloud[0]->size() << endl;

	//update resolution after downsamping
	computeResolution();
}

void CApp::_doDownSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double downsamplingSize)
{
	pcl::VoxelGrid<pcl::PointXYZ> filter;
	filter.setInputCloud(cloud);
	filter.setLeafSize(downsamplingSize * _resolution, downsamplingSize * _resolution, downsamplingSize * _resolution);
	filter.filter(*cloud);
}

//compute pointcloud resolution
void CApp::computeResolution()
{
	double res1 = _doComputeResolution(_pointcloud[0]);
	double res2 = _doComputeResolution(_pointcloud[1]);
	_resolution = (res1 + res2) / 2;
}

double CApp::_doComputeResolution(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices;
	std::vector<float> sqr_distances;
	pcl::KdTreeFLANN<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!pcl_isfinite((*cloud)[i].x))
		{
			continue;
		}
		//Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(cloud->points[i], 2, indices, sqr_distances);//return :number of neighbors found 
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}

void CApp::computeNormals()
{
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
	pcl::PointCloud<pcl::Normal>::Ptr normals_src(new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::Normal>::Ptr normals_tar(new pcl::PointCloud<pcl::Normal>());
	norm_est.setKSearch(50);
	norm_est.setInputCloud(_pointcloud[0]);
	norm_est.compute(*normals_src);
	norm_est.setInputCloud(_pointcloud[1]);
	norm_est.compute(*normals_tar);
	_normals.push_back(normals_src);
	_normals.push_back(normals_tar);
}

/**
* get HarrisKeypoint3D 
* radius means support radius
*/
void CApp::getHarrisKeypoint3D(double radius)
{
	_doGetgetHarrisKeypoint3D(radius, _pointcloud[0]);
	_doGetgetHarrisKeypoint3D(radius, _pointcloud[1]);
}

//Compute the key points and save the Index of key points
void CApp::_doGetgetHarrisKeypoint3D(double radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
{
	pcl::PointCloud<pcl::PointXYZI> result;
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI, pcl::Normal> harris;
	harris.setInputCloud(cloud_in);
	harris.setNonMaxSupression(true);
	harris.setRefine(false);
	harris.setRadius(radius * _resolution);
	harris.compute(result);
	pcl::copyPointCloud(result, *keyPoint);

	//cout << "before:" << keyPoint->size() << endl;

	//compute the index of keypoints
	std::vector<int>keyPointIdx = _computeKeyPointIndex(cloud_in, keyPoint);
	// remove invalid key points(The number of points around the key point is less than 10)
	keyPoint = _removeInvalidkeyPoint(cloud_in, keyPointIdx, keyPoint);

	//cout << "after:" << keyPoint->size() << endl;

	_key_points_idx.push_back(keyPointIdx);
	_key_points.push_back(keyPoint);
}

// remove invalid key points(The number of points around the key point is less than 10)
std::vector<int> CApp::_computeKeyPointIndex(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint)
{
	std::vector<int>keyPointIndex;
	std::vector<int>index;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<float> Dist;
	kdtree.setInputCloud(cloud_in);
	for (int i = 0; i < keyPoint->size(); i++)
	{
		index.clear();
		Dist.clear();
		kdtree.nearestKSearch(keyPoint->points[i], 1, index, Dist);
		keyPointIndex.push_back(index[0]);
	}
	return keyPointIndex;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CApp::_removeInvalidkeyPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<int> &keyPointIdx,
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr final_keyPoint(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<int>index;
	std::vector<float> Dist;
	kdtree.setInputCloud(cloud_in);
	std::vector<int>keyPointTempIdx;
	for (int i = 0; i < keyPoint->size(); i++)
	{
		kdtree.radiusSearch(cloud_in->points[keyPointIdx[i]], 15 * _resolution, index, Dist);
		if (index.size() >= 10)
		{
			keyPointTempIdx.push_back(keyPointIdx[i]);
			final_keyPoint->push_back(keyPoint->points[i]);
		}
		index.clear();
		Dist.clear();
	}
	keyPointIdx = keyPointTempIdx;
	return final_keyPoint;
}

void CApp::SHOT_compute(double sup_radius)
{
	_doSHOT_compute(_key_points_idx[0], sup_radius, _pointcloud[0]);
	_doSHOT_compute(_key_points_idx[1], sup_radius, _pointcloud[1]);
}

void CApp::_doSHOT_compute(vector<int>indices, double sup_radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	int i, j;
	pcl::PointIndicesPtr Idx = boost::shared_ptr <pcl::PointIndices>(new pcl::PointIndices());
	for (j = 0; j < indices.size(); j++)
		Idx->indices.push_back(indices[j]);
	///////
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(50);
	n.compute(*normals);
	/////////////////////////////////////////////////////////////////////////////////////////////
	pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot_est;
	shot_est.setInputCloud(cloud);
	shot_est.setInputNormals(normals);
	pcl::PointCloud<pcl::SHOT352>::Ptr shots(new pcl::PointCloud<pcl::SHOT352>());
	shot_est.setSearchMethod(tree);
	shot_est.setIndices(Idx);
	shot_est.setRadiusSearch(sup_radius * _resolution);
	shot_est.compute(*shots);
	//features.resize(shots->points.size());
	//for (i = 0; i < features.size(); i++)
	//{
	//	features[i].resize(352);
	//	for (j = 0; j < 352; j++)
	//	{
	//		features[i][j] = shots->points[i].descriptor[j];
	//	}
	//}
	_descriptor.push_back(shots);
}

void CApp::calculateTopKMatchByRatio(int k)
{
	auto descriptors_src = _descriptor[0], descriptors_tar = _descriptor[1];
	auto keyPoint_index_src = _key_points_idx[0], keyPoint_index_tar = _key_points_idx[1];
	pcl::KdTreeFLANN<pcl::SHOT352>kdtree;
	vector<int>Idx;
	vector<float>Dist;
	kdtree.setInputCloud(descriptors_tar);
	std::vector<Match_pair> match;
	Match_pair pair;
	pcl::SHOT352 n1, n2, src_i;
	for (int i = 0; i < descriptors_src->size(); i++)
	{
		kdtree.nearestKSearch(descriptors_src->points[i], 2, Idx, Dist);
		float up = sqrt(Dist[0]), down = sqrt(Dist[1]);
		pair.source_idx = keyPoint_index_src[i];
		pair.target_idx = keyPoint_index_tar[Idx[0]];
		if (down < 1e-6)continue;
		pair.ratio = 1.0 - (up / down);
		match.push_back(pair);
	}
	sort(match.begin(), match.end(), [](Match_pair m1, Match_pair m2)->bool { return m1.ratio > m2.ratio; });
	for (int i = 0; i < k; i++)
	{
		_match.push_back(match[i]);
	}
}

std::vector<Match_pair> CApp::getMatch()
{
	return _match;
}

/**
* get correct matches from initial matches 
* Mat: ground truth matrix
* correct_thresh: threshold to determine if the match is correct
*/
std::vector<int> CApp::_getCorrectMatches(Eigen::Matrix4f& Mat, double correct_thresh)
{
	int i, j;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);

	for (int i = 0; i < _match.size(); i++)
	{
		int source_idx = _match[i].source_idx;
		int target_idx = _match[i].target_idx;
		source_match_points->points.push_back(_pointcloud[0]->points[source_idx]);
		target_match_points->points.push_back(_pointcloud[1]->points[target_idx]);
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source_match_points, *source_match_points_trans, Mat);
	vector<int>ans;
	//
	for (i = 0; i < source_match_points_trans->points.size(); i++)
	{
		float X = source_match_points_trans->points[i].x - target_match_points->points[i].x;
		float Y = source_match_points_trans->points[i].y - target_match_points->points[i].y;
		float Z = source_match_points_trans->points[i].z - target_match_points->points[i].z;
		float dist = sqrt(X * X + Y * Y + Z * Z);
		if (dist < correct_thresh * _resolution)ans.push_back(1);
		else ans.push_back(0);
	}
	return ans;
}

void CApp::constructGraph(double compatibility_threshold)
{
	sac_cot::Graph graph;
	graph.nodesInitialization(_match);
	graph.computeMatiax(_pointcloud, _match, _normals, compatibility_threshold, _resolution);
	graph.computeLine();
	graph.computeCircle();
	graph.computeTriple(30);
	_circles = graph.getCircles();
	_triples = graph.getTriples();
}
void CApp::ransac(sac_cot::BaseRansac* baseRansac)
{
	if (SAC_COT *sc = dynamic_cast<SAC_COT*>(baseRansac))
	{
		sc->setCloudSrc(_pointcloud[0]);
		sc->setCloudTar(_pointcloud[1]);
		sc->setLoss("MAE");
		sc->setMatch(_match);
		sc->setResolution(_resolution);
		sc->setThreshold(7.5);
		sc->setIteratorNums(100);
		sc->setCircles(_circles);
		sc->setTriples(_triples);
		sc->ransac();
		Eigen::Matrix4f mat = sc->getMatrix();
		_matrix = mat;
		std::cout<<
			mat(0, 0) << ' ' << mat(0, 1) << ' ' << mat(0, 2) << ' ' << mat(0, 3) << ' ' << std::endl <<
			mat(1, 0) << ' ' << mat(1, 1) << ' ' << mat(1, 2) << ' ' << mat(1, 3) << ' ' << std::endl <<
			mat(2, 0) << ' ' << mat(2, 1) << ' ' << mat(2, 2) << ' ' << mat(2, 3) << ' ' << std::endl <<
			mat(3, 0) << ' ' << mat(3, 1) << ' ' << mat(3, 2) << ' ' << mat(3, 3);
	}
}