#include"ransac.h"

namespace sac_cot
{
	void SAC_COT::ransac()
	{
		int iteratorNum = getIteratorNum();
		auto match = getMatch();
		auto cloud_src = getCloudSrc(), cloud_tar = getCloudTar();
		pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
		for (int i = 0; i < match.size(); i++)
		{
			pcl::PointXYZ point_s, point_t;
			point_s = cloud_src->points[match[i].source_idx];
			point_t = cloud_tar->points[match[i].target_idx];
			source_match_points->points.push_back(point_s);
			target_match_points->points.push_back(point_t);
		}
		int size = _circles.size();
		int Rand_seed = iteratorNum;
		double score = -1;
		for (int i = 0; i < iteratorNum; i++)
		{
			Rand_seed++;
			int matchIdx1, matchIdx2, matchIdx3;
			sac_cot::utils::Rand(Rand_seed, match.size(), matchIdx1, matchIdx2, matchIdx3);
			pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
			int corr1, corr2, corr3;
			if (i >= size)
			{
				corr1 = _triples[i - size].index1;
				corr2 = _triples[i - size].index2;
				corr3 = _triples[i - size].index3;
			}
			else
			{
				corr1 = _circles[i].index1;
				corr2 = _circles[i].index2;
				corr3 = _circles[i].index3;
			}
			point_s1 = cloud_src->points[match[corr1].source_idx];
			point_s2 = cloud_src->points[match[corr2].source_idx];
			point_s3 = cloud_src->points[match[corr3].source_idx];
			point_t1 = cloud_tar->points[match[corr1].target_idx];
			point_t2 = cloud_tar->points[match[corr2].target_idx];
			point_t3 = cloud_tar->points[match[corr3].target_idx];

			Eigen::Matrix4f matIterator = Eigen::Matrix4f::Identity();
			RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, matIterator);
			double scoreIterator = Score_est(source_match_points, target_match_points, matIterator);
			if (scoreIterator > score)
			{
				score = scoreIterator;
				setMatrix(matIterator);
			}
		}
	}
	void SAC_COT::setCircles(std::vector<Circle> circles) 
	{ 
		_circles = circles; 
	}
	void SAC_COT::setTriples(std::vector<Triple> triples) 
	{ 
		_triples = triples;
	}
	std::vector<Circle>SAC_COT::getCircles()
	{ 
		return _circles;
	}
	std::vector<Triple>SAC_COT::getTriples() 
	{ 
		return _triples; 
	}

	BaseRansac::BaseRansac(){}
	void BaseRansac::setIteratorNums(int num) 
	{
		_iterator_nums = num; 
	}
	void BaseRansac::setThreshold(double threshold) 
	{ 
		_threshold = threshold; 
	}
	void BaseRansac::setResolution(double resolution)
	{ 
		_resolution = resolution; 
	}
	void BaseRansac::setLoss(std::string loss) 
	{ 
		_loss = loss; 
	}
	void BaseRansac::setMatch(std::vector<Match_pair>match) 
	{ 
		_match = match; 
	}
	void BaseRansac::setCloudSrc(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src) 
	{ 
		_cloud_src = cloud_src;
	}
	void BaseRansac::setCloudTar(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar) 
	{ 
		_cloud_tar = cloud_tar;
	}
	void BaseRansac::setMatrix(Eigen::Matrix4f mat) 
	{ 
		_matrix = mat; 
	}

	int BaseRansac::getIteratorNum() 
	{
		return _iterator_nums; 
	}
	double BaseRansac::getThreshold() 
	{ 
		return _threshold;
	}
	double BaseRansac::getResolution() 
	{ 
		return _resolution;
	}
	Eigen::Matrix4f BaseRansac::getMatrix() 
	{ 
		return _matrix; 
	}
	std::string BaseRansac::getLoss() 
	{
		return _loss; 
	}
	std::vector<Match_pair> BaseRansac::getMatch() 
	{ 
		return _match; 
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr BaseRansac::getCloudSrc() 
	{ 
		return _cloud_src; 
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr BaseRansac::getCloudTar() 
	{ 
		return _cloud_tar;
	}

	void BaseRansac::RANSAC_trans_est(pcl::PointXYZ& point_s1, pcl::PointXYZ& point_s2, pcl::PointXYZ& point_s3,
		pcl::PointXYZ& point_t1, pcl::PointXYZ& point_t2, pcl::PointXYZ& point_t3, Eigen::Matrix4f& mat)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr LRF_source(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr LRF_target(new pcl::PointCloud<pcl::PointXYZ>);

		LRF_source->points.push_back(point_s1); LRF_source->points.push_back(point_s2); LRF_source->points.push_back(point_s3);//LRF_source->points.push_back(s_4);
		LRF_target->points.push_back(point_t1); LRF_target->points.push_back(point_t2); LRF_target->points.push_back(point_t3);//LRF_source->points.push_back(t_4);
		//
		pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> SVD;
		SVD.estimateRigidTransformation(*LRF_source, *LRF_target, mat);
	}

	double BaseRansac::Score_est(pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points,
		pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points, Eigen::Matrix4f mat)
	{
		int i, j;
		double threshold = _threshold * _resolution;
		pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points_trans(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*source_match_points, *source_match_points_trans, mat);
		double score = 0;
		if (_loss.compare("Inlier") == 0)
		{
			score = RANSAC_inliers(source_match_points, target_match_points, mat, threshold);
		}
		else
		{
			for (i = 0; i < source_match_points_trans->points.size(); i++)
			{
				double X = source_match_points_trans->points[i].x - target_match_points->points[i].x;
				double Y = source_match_points_trans->points[i].y - target_match_points->points[i].y;
				double Z = source_match_points_trans->points[i].z - target_match_points->points[i].z;
				double dist = sqrt(X * X + Y * Y + Z * Z), temp_score;
				if (_loss.compare("MAE") == 0)
				{
					if (dist < threshold)
					{
						temp_score = (threshold - dist) / threshold;
						score += temp_score;
					}
				}
				else if (_loss.compare("MSE") == 0)
				{
					if (dist < threshold)
					{
						temp_score = (dist - threshold) * (dist - threshold) / (threshold * threshold);
						score += temp_score;
					}
				}
				else if (_loss.compare("LOG-COSH") == 0)
				{
					if (dist < threshold)
					{
						temp_score = log(cosh(threshold - dist)) / log(cosh(threshold));
						score += temp_score;
					}
				}
				else if (_loss.compare("QUANTILE") == 0)
				{
					if (dist < threshold)
					{
						temp_score = 0.9*(threshold - dist) / threshold;
					}
					else temp_score = 0.1*(dist - threshold) / dist;
					score += temp_score;
				}
				else if (_loss.compare("-QUANTILE") == 0)
				{
					if (dist < threshold) temp_score = 0.9*(threshold - dist) / threshold;
					else temp_score = 0.1*(threshold - dist) / dist;
					score += temp_score;
				}
				else if (_loss.compare("EXP") == 0)
				{
					if (dist < threshold)
					{
						temp_score = exp(-(pow(dist, 2) / (2 * pow(threshold, 2))));
						score += temp_score;
					}
				}
			}
		}
		return score;
	}

	double BaseRansac::RANSAC_inliers(pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points, pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points,
		Eigen::Matrix4f& mat, double correct_thresh)
	{
		int i, j;
		pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points_trans(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*source_match_points, *source_match_points_trans, mat);
		//
		double N = 0;
		for (i = 0; i < source_match_points_trans->points.size(); i++)
		{
			double X = source_match_points_trans->points[i].x - target_match_points->points[i].x;
			double Y = source_match_points_trans->points[i].y - target_match_points->points[i].y;
			double Z = source_match_points_trans->points[i].z - target_match_points->points[i].z;
			double dist = sqrt(X * X + Y * Y + Z * Z);
			if (dist < correct_thresh)N++;
		}
		return N;
	}
}