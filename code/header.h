#include<iostream>
#include<cmath>
#include<pcl/point_types.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<vector>
#include<set>
#include<algorithm>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <stdio.h>
#include <vector>
#include <pcl\point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>   //法线
#include <pcl/features/shot_omp.h>    //描述子
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/registration/transforms.h>
#include <boost/thread/thread.hpp>
#include <pcl/registration/transformation_estimation_svd.h>
#include<pcl/filters/voxel_grid.h>
#include <time.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include<iostream>
#include<pcl/io/pcd_io.h>
#include <pcl/point_types.h>  
#define BOOST_TYPEOF_EMULATION
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/fpfh.h>
#include <pcl/keypoints/harris_3d.h>
#include<pcl/keypoints/iss_3d.h>
#include <pcl/console/time.h> 
#include <pcl/features/normal_3d_omp.h>  
#include <pcl/features/spin_image.h>
#include<algorithm>
#ifndef _PAIRWISEREG_EVAL_H_ 
#define _PAIRWISEREG_EVAL_H_
#define Threshold_Radius 7.5
#define averDistanceInit 2e20
using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointXYZ PointInT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
typedef struct {
	int index1;
	int index2;
	int index3;
	int degree;
	double compatibility_dist;
	double area;
}Circle;
typedef struct {
	int index1;
	int index2;
	int degree;
	double compatibility_dist;
}Line;
typedef struct {
	int index1;
	int index2;
	int index3;
	int degree;
}Triple;
typedef struct {
	int index;//第index根匹配
	int degree;//此匹配的度
}Node;
typedef struct {
	float x;
	float y;
	float z;
}Vertex;
typedef struct {
	int pointID1;
	int pointID2;
	int pointID3;
	float center_x;
	float center_y;
	float center_z;
}Triangle_with_center;
typedef struct {
	int pointID;
	Vertex x_axis;
	Vertex y_axis;
	Vertex z_axis;
}LRF;
typedef struct {
	int iID;
	int jID;
	int N;
	Eigen::Matrix4f Mat;
	PointCloudPtr cloud_i;
	PointCloudPtr cloud_j;
}PairLoop;
typedef struct {
	PointCloudPtr cloud;
	std::vector<int> Idx;
	std::vector<LRF> LRFs;
	std::vector<std::vector<float>> features;
	float mesh_res;
}Data_with_features;
typedef struct {
	int R;
	int G;
	int B;
}RGB;
typedef struct {
	float x;
	float y;
	float z;
	float dist;
	float angle_to_axis;
}Vertex_d_ang;
typedef struct {
	int source_idx;
	int target_idx;
	LRF source_LRF;
	LRF target_LRF;
	float ratio;
	float dist;//nearest neighbor distance ratio 
}Match_pair;
typedef struct {
	float M[4][4];
}TransMat;
typedef struct {
	int SourceID;
	int TargetID;
	Eigen::Matrix4f Mat_coarse;
	Eigen::Matrix4f Mat_fine;
	float rot_error;
	float trans_error;
	float RMSE;
	float time;
	bool Reg_judge;
}Pairwise_Result;
void boost_rand(int seed, int start, int end, int rand_num, std::vector<int>& idx);
void Rand_3(int seed, int scale, int& output1, int& output2, int& output3);
void Rand_2(int seed, int scale, int& output1, int& output2);
void Rand_1(int seed, int scale, int& output);
int XYZ_Read(string Filename, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
double getCompatibility(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j, pcl::Normal &ns_i,
	pcl::Normal &ns_j, pcl::Normal &nt_i, pcl::Normal &nt_j, float resolution);
double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud);
PointCloudPtr downSampling(PointCloudPtr cloud_in, float resolution, float downSize);
pcl::PointCloud<pcl::PointXYZ>::Ptr getHarrisKeypoint3D(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
	float radius, float resolution);
pcl::PointIndicesPtr getIndexofKeyPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint, std::vector<int>& keyPoint_Idx);
PointCloudPtr removeInvalidkeyPoint(PointCloudPtr cloud_in, vector<int> &keyPointIdx, PointCloudPtr keyPoint, float resolution);
pcl::PointCloud<pcl::SHOT352>::Ptr SHOT_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
	vector<int>&indices, float sup_radius, vector<vector<float>>&features);
vector<Match_pair> getTopKCorresByRatio(pcl::PointCloud<pcl::SHOT352>::Ptr descriptors_src,
	pcl::PointCloud<pcl::SHOT352>::Ptr descriptors_tar, vector<int> keyPoint_index_src, vector<int> keyPoint_index_tar, int k);
float Score_est(pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points,
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points, PointCloudPtr cloud_source, 
	PointCloudPtr cloud_target, Eigen::Matrix4f Mat, float thresh, string loss, float resolution);
int RANSAC_score(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
float RMSE_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target,
	Eigen::Matrix4f&Mat_est, Eigen::Matrix4f&Mat_GT, float mr);
vector<Circle> getCircle(vector<vector<int>>matrix, int n, vector<Node>& nodes, vector<vector<double>>& compatibility_matrix);
vector<Line> getLine(vector<vector<int>>&matrix, vector<Node>& nodes, vector<vector<double>>& compatibility_matrix, int n);
int GuideSampling_score(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Circle> circles, vector<Triple>triples, vector<Match_pair> match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
int CG_SAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Line>lines, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
vector<int> getCorrectCorrs(PointCloudPtr source_match_points, PointCloudPtr target_match_points, Eigen::Matrix4f& Mat, float correct_thresh);
void computeMatiax(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Match_pair>  match, pcl::PointCloud<pcl::Normal>::Ptr normals_src, 
	pcl::PointCloud<pcl::Normal>::Ptr normals_tar, double threshod, vector<Node>& nodes, vector<vector<int>>& adjacent_matrix, 
	vector<vector<double>>& compatibility_matrix, float resolution);
vector<Triple>getTriple(vector<Node>& nodes, int topK);
float getAverDistance(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, Eigen::Matrix4f& Mat, float resolution);
float HuberDistance(PointCloudPtr source_match_points, PointCloudPtr target_match_points, Eigen::Matrix4f& Mat, float resolution);
int FarSampleRANSAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
int SAC_IA(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
int OnePointRANSAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
int OSAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
int TwoSAC_GC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
int GC1SAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
float RANSAC_overlap(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, Eigen::Matrix4f& Mat, float alpha, float resolution);
void TOLDI_LRF_for_cloud_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<int> indices, float sup_radius, vector<LRF>& Cloud_LRF);
pcl::PointCloud<pcl::Histogram<153> >::Ptr SpinImage_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
	vector<int>&indices, float sup_radius, pcl::PointCloud<pcl::Normal>::Ptr normals);
void pointLFSH(float r, int bin_num, Vertex &searchPoint, Vertex& n, pcl::PointCloud<pcl::PointXYZ>::Ptr&neigh, pcl::PointCloud<pcl::Normal>::Ptr&sphere_normals,
	vector<float> &histogram);
void LFSH_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<int> indices, float sup_radius, int bin_num, vector<vector<float>>&Histograms);
pcl::PointCloud<pcl::SHOT352>::Ptr SpinImageToShot(pcl::PointCloud<pcl::Histogram<153> >::Ptr SpinImage);
pcl::PointCloud<pcl::SHOT352>::Ptr LFSHToShot(vector<vector<float>>LFSH);
pcl::PointCloud<pcl::PointXYZ>::Ptr getISS3dKeypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, float resolution);
vector<Match_pair> getTopKCorresByRatio(pcl::PointCloud<pcl::SHOT352>::Ptr descriptors_src,
	pcl::PointCloud<pcl::SHOT352>::Ptr descriptors_tar, vector<int> keyPoint_index_src, vector<int> keyPoint_index_tar, float ratio);
int getCorrectCorrespondences(vector<Match_pair> match, vector<int>&corrcet_match, Eigen::Matrix4f Mat, float thresh,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar);
vector<Circle> getCliqueCircle(vector<int>maxClique, vector<vector<double>>& compatibility_matrix);
int CliqueSampling_score(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Circle> circles, vector<Triple>triples, vector<Match_pair> Match,
	float resolution, int  Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
vector<Match_pair> getTopKCorresByRatio1344(pcl::PointCloud<pcl::SHOT1344>::Ptr descriptors_src,
	pcl::PointCloud<pcl::SHOT1344>::Ptr descriptors_tar, vector<int> keyPoint_index_src, vector<int> keyPoint_index_tar, float ratio);
float getOverlapRate(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_i, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_j, Eigen::Matrix4f GroundTruth, float resolution);
bool compairCircleByDgree(const Circle& c1, const Circle& c2);
bool compairCircleByCompatibilityDist(const Circle& c1, const Circle& c2);
bool compairCircleByArea(const Circle& c1, const Circle& c2);
bool compairNodeBydegree(const Node& n1, const Node& n2);
bool compairTripleBydegree(const Triple& t1, const Triple& t2);
bool compairLineByCompatibilityDist(const Line& l1, const Line& l2);
#endif