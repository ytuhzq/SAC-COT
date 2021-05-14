#pragma once
#include"CApp.h"
#include"utils.h"
#include <pcl/registration/transformation_estimation_svd.h>
namespace sac_cot
{
	class BaseRansac
	{
	public:
		BaseRansac();

		virtual void ransac() {};

		void setIteratorNums(int num);
		void setThreshold(double threshold);
		void setResolution(double resolution);
		void setLoss(std::string loss);
		void setMatch(std::vector<Match_pair>match);
		void setCloudSrc(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src);
		void setCloudTar(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar);
		void setMatrix(Eigen::Matrix4f mat);

		int getIteratorNum();
		double getThreshold();
		double getResolution();
		Eigen::Matrix4f getMatrix();
		std::string getLoss();
		std::vector<Match_pair> getMatch();
		pcl::PointCloud<pcl::PointXYZ>::Ptr getCloudSrc();
		pcl::PointCloud<pcl::PointXYZ>::Ptr getCloudTar();

		void RANSAC_trans_est(pcl::PointXYZ& point_s1, pcl::PointXYZ& point_s2, pcl::PointXYZ& point_s3,
			pcl::PointXYZ& point_t1, pcl::PointXYZ& point_t2, pcl::PointXYZ& point_t3, Eigen::Matrix4f& mat);

		double Score_est(pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points,
			pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points, Eigen::Matrix4f mat);

		double RANSAC_inliers(pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points, pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points,
			Eigen::Matrix4f& mat, double correct_thresh);

	private:
		int _iterator_nums;
		double _threshold;
		double _resolution;
		std::string _loss;
		Eigen::Matrix4f _matrix;
		std::vector<Match_pair> _match;
		pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud_src;
		pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud_tar;
	};

	class SAC_COT : public BaseRansac{
	public:
		SAC_COT() : BaseRansac(){}
		void setCircles(std::vector<Circle> circles);
		void setTriples(std::vector<Triple> triples);
		std::vector<Circle> getCircles();
		std::vector<Triple> getTriples();

		void ransac();

	private:
		std::vector<Circle> _circles;
		std::vector<Triple> _triples;
	};
}

