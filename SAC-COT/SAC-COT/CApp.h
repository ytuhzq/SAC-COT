#pragma once
#include<string>
#include<vector>
#include <pcl/common/eigen.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>   //·¨Ïß
#include <pcl/keypoints/harris_3d.h>
#include <pcl/features/shot_omp.h> 
#include"geometry.h"
#include"ransac.h"

namespace sac_cot
{
class BaseRansac;
class CApp
{
public:
	CApp(){};

	int XYZ_Read(const char* Filename);
	int XYZ_Save(const char* path, const char* src, const char* tar);
	void LoadPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
	void computeNormals();
	void computeResolution();
	void downSampling(double downsamplingSize = 2.0);
	void getHarrisKeypoint3D(double radius = 2.0);
	void SHOT_compute(double sup_radius = 15);
	void calculateTopKMatchByRatio(int k);
	std::vector<Match_pair>getMatch();
	void constructGraph(double compatibility_threshold = 0.9);
	void ransac(BaseRansac* baseRansac);
	
private:
	double _resolution;
	std::vector<Match_pair> _match;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>_pointcloud;
	std::vector<pcl::PointCloud<pcl::Normal>::Ptr>_normals;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>_key_points;
	std::vector<std::vector<int>>_key_points_idx;
	std::vector<pcl::PointCloud<pcl::SHOT352>::Ptr>_descriptor;
	std::vector<Circle> _circles;
	std::vector<Triple>_triples;
	Eigen::Matrix4f _matrix;

	void _doGetgetHarrisKeypoint3D(double radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in);
	void _doDownSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double downsamplingSize);
	double _doComputeResolution(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
	std::vector<int> _computeKeyPointIndex(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint);
	pcl::PointCloud<pcl::PointXYZ>::Ptr _removeInvalidkeyPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<int> &keyPointIdx,
		pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint);
	void _doSHOT_compute(std::vector<int>indices, double sup_radius, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
	std::vector<int> _getCorrectMatches(Eigen::Matrix4f& Mat, double correct_thresh);
};

}