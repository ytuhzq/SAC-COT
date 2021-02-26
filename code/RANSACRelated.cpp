#include"header.h"
void boost_rand(int seed, int start, int end, int rand_num, std::vector<int>& idx)
{
	boost::mt19937 engine(seed);
	boost::uniform_int<> distribution(start, end);
	boost::variate_generator<boost::mt19937, boost::uniform_int<> > myrandom(engine, distribution);
	for (int i = 0; i < rand_num; i++)
		idx.push_back(myrandom());
}
void Rand_3(int seed, int scale, int& output1, int& output2, int& output3)
{
	std::vector<int> result;
	int start = 0;
	int end = scale - 1;
	boost_rand(seed, start, end, scale, result);
	output1 = result[0];
	output2 = result[1];
	output3 = result[2];
}
void Rand_2(int seed, int scale, int& output1, int& output2)
{
	std::vector<int> result;
	int start = 0;
	int end = scale - 1;
	boost_rand(seed, start, end, scale, result);
	output1 = result[0];
	output2 = result[1];
}
void Rand_1(int seed, int scale, int& output)
{
	std::vector<int> result;
	int start = 0;
	int end = scale - 1;
	boost_rand(seed, start, end, scale, result);
	output = result[0];
}

//Hypothesis quality estimation
float RANSAC_inliers(PointCloudPtr source_match_points, PointCloudPtr target_match_points, Eigen::Matrix4f& Mat, float correct_thresh)
{
	int i, j;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source_match_points, *source_match_points_trans, Mat);
	//
	float N = 0;
	for (i = 0; i < source_match_points_trans->points.size(); i++)
	{
		float X = source_match_points_trans->points[i].x - target_match_points->points[i].x;
		float Y = source_match_points_trans->points[i].y - target_match_points->points[i].y;
		float Z = source_match_points_trans->points[i].z - target_match_points->points[i].z;
		float dist = sqrt(X * X + Y * Y + Z * Z);
		if (dist < correct_thresh)N++;
	}
	return N;
}

vector<int> getCorrectCorrs(PointCloudPtr source_match_points, PointCloudPtr target_match_points, Eigen::Matrix4f& Mat, float correct_thresh)
{
	int i, j;
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
		if (dist < correct_thresh)ans.push_back(1);
		else ans.push_back(0);
	}
	return ans;
}


//Hypothesis generation
void RANSAC_trans_est(pcl::PointXYZ& point_s1, pcl::PointXYZ& point_s2, pcl::PointXYZ& point_s3,
	pcl::PointXYZ& point_t1, pcl::PointXYZ& point_t2, pcl::PointXYZ& point_t3, Eigen::Matrix4f& Mat)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr LRF_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr LRF_target(new pcl::PointCloud<pcl::PointXYZ>);

	LRF_source->points.push_back(point_s1); LRF_source->points.push_back(point_s2); LRF_source->points.push_back(point_s3);//LRF_source->points.push_back(s_4);
	LRF_target->points.push_back(point_t1); LRF_target->points.push_back(point_t2); LRF_target->points.push_back(point_t3);//LRF_source->points.push_back(t_4);
	//
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> SVD;
	SVD.estimateRigidTransformation(*LRF_source, *LRF_target, Mat);
}

float Score_est(pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points,
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points, PointCloudPtr cloud_source, 
	PointCloudPtr cloud_target, Eigen::Matrix4f Mat, float thresh, string loss, float resolution)
{
	int i, j;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source_match_points, *source_match_points_trans, Mat);
	float score = 0;
	if (loss.compare("Inlier") == 0)
	{
		score = RANSAC_inliers(source_match_points, target_match_points, Mat, thresh);
	}
	else
	{
		for (i = 0; i < source_match_points_trans->points.size(); i++)
		{
			float X = source_match_points_trans->points[i].x - target_match_points->points[i].x;
			float Y = source_match_points_trans->points[i].y - target_match_points->points[i].y;
			float Z = source_match_points_trans->points[i].z - target_match_points->points[i].z;
			float dist = sqrt(X * X + Y * Y + Z * Z), temp_score;
			if (loss.compare("MAE") == 0)
			{
				if (dist < thresh)
				{
					temp_score = (thresh - dist) / thresh;
					score += temp_score;
				}
			}
			else if (loss.compare("MSE") == 0)
			{
				if (dist < thresh)
				{
					temp_score = (dist - thresh) * (dist - thresh) / (thresh * thresh);
					score += temp_score;
				}
			}
			else if (loss.compare("LOG-COSH") == 0)
			{
				if (dist < thresh)
				{
					temp_score = log(cosh(thresh - dist)) / log(cosh(thresh));
					score += temp_score;
				}
			}
			else if (loss.compare("QUANTILE") == 0)
			{
				if (dist < thresh)
				{
					temp_score = 0.9*(thresh - dist) / thresh;
				}
				else temp_score = 0.1*(dist - thresh) / dist;
				score += temp_score;
			}
			else if (loss.compare("-QUANTILE") == 0)
			{
				if (dist < thresh) temp_score = 0.9*(thresh - dist) / thresh;
				else temp_score = 0.1*(thresh - dist) / dist;
				score += temp_score;
			}
			else if (loss.compare("EXP") == 0)
			{
				if (dist < thresh)
				{
					temp_score = exp(-(pow(dist, 2) / (2 * pow(thresh, 2))));
					score += temp_score;
				}
			}
		}
	}
	return score;
}
int RANSAC_score(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{
	Mat = Eigen::Matrix4f::Identity();
	float RANSAC_inlier_judge_thresh = threshold * resolution;
	float score = -999999;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ point_s, point_t;
		point_s = cloud_source->points[Match[i].source_idx];
		point_t = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(point_s);
		target_match_points->points.push_back(point_t);
	}
	//
	int Iterations = _Iterations;
	int Rand_seed = Iterations;
	while (Iterations)
	{
		Rand_seed++;
		int Match_Idx1, Match_Idx2, Match_Idx3;
		Rand_3(Rand_seed, Match.size(), Match_Idx1, Match_Idx2, Match_Idx3);
		pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		point_s1 = cloud_source->points[Match[Match_Idx1].source_idx];
		point_s2 = cloud_source->points[Match[Match_Idx2].source_idx];
		point_s3 = cloud_source->points[Match[Match_Idx3].source_idx];
		point_t1 = cloud_target->points[Match[Match_Idx1].target_idx];
		point_t2 = cloud_target->points[Match[Match_Idx2].target_idx];
		point_t3 = cloud_target->points[Match[Match_Idx3].target_idx];
		//
		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		float score_iter = Score_est(source_match_points, target_match_points, cloud_source, cloud_target, Mat_iter, RANSAC_inlier_judge_thresh, loss, resolution);
		if (score_iter > score)
		{
			score = score_iter;
			Mat = Mat_iter;
		}
		Iterations--;
	}
	return 1;
}

int GuideSampling_score(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Circle> circles, vector<Triple>triples, vector<Match_pair> Match,
	float resolution, int  Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{

	Mat = Eigen::Matrix4f::Identity();
	float RANSAC_inlier_judge_thresh = threshold * resolution;
	float score = -999999;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ point_s, point_t;
		point_s = cloud_source->points[Match[i].source_idx];
		point_t = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(point_s);
		target_match_points->points.push_back(point_t);
	}
	int size = circles.size();
	int Rand_seed = Iterations;
	for (int i = 0; i < Iterations; i++)
	{
		Rand_seed++;
		int Match_Idx1, Match_Idx2, Match_Idx3;
		Rand_3(Rand_seed, Match.size(), Match_Idx1, Match_Idx2, Match_Idx3);
		pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		//corr1表示环中的第一根匹配索引
		int corr1, corr2, corr3;
		if (i >= size)
		{
			if (i - size > triples.size())cout << triples.size() << endl;
			corr1 = triples[i - size].index1;
			corr2 = triples[i - size].index2;
			corr3 = triples[i - size].index3;
		}
		else
		{
			corr1 = circles[i].index1;
			corr2 = circles[i].index2;
			corr3 = circles[i].index3;
		}
		//match[corr1]表示环中第一根匹配
		point_s1 = cloud_source->points[Match[corr1].source_idx];
		point_s2 = cloud_source->points[Match[corr2].source_idx];
		point_s3 = cloud_source->points[Match[corr3].source_idx];
		point_t1 = cloud_target->points[Match[corr1].target_idx];
		point_t2 = cloud_target->points[Match[corr2].target_idx];
		point_t3 = cloud_target->points[Match[corr3].target_idx];

		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		float score_iter = Score_est(source_match_points, target_match_points, cloud_source, cloud_target,Mat_iter, RANSAC_inlier_judge_thresh, loss, resolution);
		//cout << "score_iter=" << score_iter << endl;
		if (score_iter > score)
		{
			score = score_iter;
			Mat = Mat_iter;
		}
	}
	//system("pause");
	return 1;
}

int CliqueSampling_score(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Circle> circles, vector<Triple>triples, vector<Match_pair> Match,
	float resolution, int  Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{

	Mat = Eigen::Matrix4f::Identity();
	float RANSAC_inlier_judge_thresh = threshold * resolution;
	float score = -999999;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ point_s, point_t;
		point_s = cloud_source->points[Match[i].source_idx];
		point_t = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(point_s);
		target_match_points->points.push_back(point_t);
	}
	int size = circles.size();
	int Rand_seed = Iterations;
	for (int i = 0; i < Iterations; i++)
	{
		Rand_seed++;
		int Match_Idx1, Match_Idx2, Match_Idx3;
		Rand_3(Rand_seed, Match.size(), Match_Idx1, Match_Idx2, Match_Idx3);
		pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		//corr1表示环中的第一根匹配索引
		int corr1, corr2, corr3;
		if (i >= size)
		{
			if (i - size > triples.size())cout << triples.size() << endl;
			corr1 = triples[i - size].index1;
			corr2 = triples[i - size].index2;
			corr3 = triples[i - size].index3;
		}
		else
		{
			corr1 = circles[i].index1;
			corr2 = circles[i].index2;
			corr3 = circles[i].index3;
		}
		//match[corr1]表示环中第一根匹配
		point_s1 = cloud_source->points[Match[corr1].source_idx];
		point_s2 = cloud_source->points[Match[corr2].source_idx];
		point_s3 = cloud_source->points[Match[corr3].source_idx];
		point_t1 = cloud_target->points[Match[corr1].target_idx];
		point_t2 = cloud_target->points[Match[corr2].target_idx];
		point_t3 = cloud_target->points[Match[corr3].target_idx];
		//cout << Match[corr1].source_idx << " " << Match[corr2].source_idx << " " << Match[corr3].source_idx << endl;
		//cout << Match[corr1].target_idx << " " << Match[corr2].target_idx << " " << Match[corr3].target_idx << endl;
		//system("pause");

		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		float score_iter = Score_est(source_match_points, target_match_points, cloud_source, cloud_target, Mat_iter, RANSAC_inlier_judge_thresh, loss, resolution);
		//cout << "score_iter=" << score_iter << endl;
		if (score_iter > score)
		{
			score = score_iter;
			Mat = Mat_iter;
		}
	}
	//system("pause");
	return 1;
}

int FarSampleRANSAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{
	Mat = Eigen::Matrix4f::Identity();
	float RANSAC_inlier_judge_thresh = Threshold_Radius * resolution;
	float score = -999999;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ point_s, point_t;
		point_s = cloud_source->points[Match[i].source_idx];
		point_t = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(point_s);
		target_match_points->points.push_back(point_t);
	}
	//
	int Iterations = _Iterations;
	int Rand_seed = Iterations;
	while (Iterations)
	{
		Rand_seed++;
		int Match_Idx1, Match_Idx2, Match_Idx3;
		bool flag = true;
		pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		while (flag)
		{
			Rand_seed++;//防止某次产生的伪随机数不符合要求，而随机种子不变，导致生成相同的伪随机数
			Rand_3(Rand_seed, Match.size(), Match_Idx1, Match_Idx2, Match_Idx3);
			point_s1 = cloud_source->points[Match[Match_Idx1].source_idx];
			point_s2 = cloud_source->points[Match[Match_Idx2].source_idx];
			point_s3 = cloud_source->points[Match[Match_Idx3].source_idx];
			point_t1 = cloud_target->points[Match[Match_Idx1].target_idx];
			point_t2 = cloud_target->points[Match[Match_Idx2].target_idx];
			point_t3 = cloud_target->points[Match[Match_Idx3].target_idx];
			float d1 = sqrt(pow(point_s1.x - point_s2.x, 2) + pow(point_s1.y - point_s2.y, 2) + pow(point_s1.z - point_s2.z, 2));
			float d2 = sqrt(pow(point_s1.x - point_s3.x, 2) + pow(point_s1.y - point_s3.y, 2) + pow(point_s1.z - point_s3.z, 2));
			float d3 = sqrt(pow(point_s2.x - point_s3.x, 2) + pow(point_s2.y - point_s3.y, 2) + pow(point_s2.z - point_s3.z, 2));
			if (d1 > Threshold_Radius * 2 * resolution && d2 > Threshold_Radius * 2 * resolution && d3 > Threshold_Radius * 2 * resolution)
				flag = false;
		}
		//
		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		float score_iter = Score_est(source_match_points, target_match_points, cloud_source, cloud_target, Mat_iter, RANSAC_inlier_judge_thresh, loss, resolution);
		if (score_iter > score)
		{
			score = score_iter;
			Mat = Mat_iter;
		}
		Iterations--;
	}
	return 1;
}

int SAC_IA(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{
	Mat = Eigen::Matrix4f::Identity();
	int Iterations = _Iterations;
	float RANSAC_inlier_judge_thresh = Threshold_Radius * resolution;
	float score = 2e20;
	//float score = -9999;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ point_s, point_t;
		point_s = cloud_source->points[Match[i].source_idx];
		point_t = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(point_s);
		target_match_points->points.push_back(point_t);
	}
	int Rand_seed = Iterations;
	while (Iterations)
	{
		Rand_seed++;
		int Match_Idx1, Match_Idx2, Match_Idx3;
		bool flag = true;
		pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		//while (flag)//加上限制随机匹配的距离，配准效果有了很大提高
		//{
			Rand_seed++;//防止某次产生的伪随机数不符合要求，而随机种子不变，导致生成相同的伪随机数
			Rand_3(Rand_seed, Match.size(), Match_Idx1, Match_Idx2, Match_Idx3);
			point_s1 = cloud_source->points[Match[Match_Idx1].source_idx];
			point_s2 = cloud_source->points[Match[Match_Idx2].source_idx];
			point_s3 = cloud_source->points[Match[Match_Idx3].source_idx];
			point_t1 = cloud_target->points[Match[Match_Idx1].target_idx];
			point_t2 = cloud_target->points[Match[Match_Idx2].target_idx];
			point_t3 = cloud_target->points[Match[Match_Idx3].target_idx];
		//	float d1 = sqrt(pow(point_s1.x - point_s2.x, 2) + pow(point_s1.y - point_s2.y, 2) + pow(point_s1.z - point_s2.z, 2));
		//	float d2 = sqrt(pow(point_s1.x - point_s3.x, 2) + pow(point_s1.y - point_s3.y, 2) + pow(point_s1.z - point_s3.z, 2));
		//	float d3 = sqrt(pow(point_s2.x - point_s3.x, 2) + pow(point_s2.y - point_s3.y, 2) + pow(point_s2.z - point_s3.z, 2));
		//	if (d1 > Threshold_Radius * 2 * resolution && d2 > Threshold_Radius * 2 * resolution && d3 > Threshold_Radius * 2 * resolution)
		//		flag = false;
		//}
		//
		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		//float score_iter = Score_est(source_match_points, target_match_points, cloud_source, cloud_target, Mat_iter, RANSAC_inlier_judge_thresh, loss, resolution);
		float score_iter = HuberDistance(source_match_points, target_match_points, Mat_iter, resolution);
		if (score_iter < score)
		{
			score = score_iter;
			Mat = Mat_iter;
		}
		Iterations--;
	}
	return 1;
}
int OnePointRANSAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{
	Mat = Eigen::Matrix4f::Identity();
	float RANSAC_inlier_judge_thresh = Threshold_Radius * resolution;
	float score = -999999;
	int Iterations = _Iterations;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ points, pointt;
		points = cloud_source->points[Match[i].source_idx];
		pointt = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(points);
		target_match_points->points.push_back(pointt);
	}
	int Rand_seed = Iterations;
	while (Iterations)
	{
		Rand_seed++;
		int Match_Idx;
		Rand_1(Rand_seed, Match.size(), Match_Idx);
		//cout << Match_Idx << " ";
		pcl::PointXYZ point_s, point_t, point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		point_s = cloud_source->points[Match[Match_Idx].source_idx];
		point_t = cloud_target->points[Match[Match_Idx].target_idx];
		LRF LRF_src = Match[Match_Idx].source_LRF;
		LRF LRF_tar = Match[Match_Idx].target_LRF;

		point_s1.x = LRF_src.x_axis.x + point_s.x; point_s1.y = LRF_src.x_axis.y + point_s.y; point_s1.z = LRF_src.x_axis.z + point_s.z;
		point_s2.x = LRF_src.y_axis.x + point_s.x; point_s2.y = LRF_src.y_axis.y + point_s.y; point_s2.z = LRF_src.y_axis.z + point_s.z;
		point_s3.x = LRF_src.z_axis.x + point_s.x; point_s3.y = LRF_src.z_axis.y + point_s.y; point_s3.z = LRF_src.z_axis.z + point_s.z;

		point_t1.x = LRF_tar.x_axis.x + point_t.x; point_t1.y = LRF_tar.x_axis.y + point_t.y; point_t1.z = LRF_tar.x_axis.z + point_t.z;
		point_t2.x = LRF_tar.y_axis.x + point_t.x; point_t2.y = LRF_tar.y_axis.y + point_t.y; point_t2.z = LRF_tar.y_axis.z + point_t.z;
		point_t3.x = LRF_tar.z_axis.x + point_t.x; point_t3.y = LRF_tar.z_axis.y + point_t.y; point_t3.z = LRF_tar.z_axis.z + point_t.z;
		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		float score_iter = Score_est(source_match_points, target_match_points, cloud_source, cloud_target, Mat_iter, RANSAC_inlier_judge_thresh, loss, resolution);
		if (score_iter > score)
		{
			//cout << "Iterations="<< Iterations<<"inliers_iter=" << inliers_iter << endl;
			//std::cout << "acc=" << (float)((float)inliers_iter / (float)Match.size()) << endl;
			score = score_iter;
			Mat = Mat_iter;
		}
		Iterations--;
	}
	return 1;
}
int OSAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ points, pointt;
		points = cloud_source->points[Match[i].source_idx];
		pointt = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(points);
		target_match_points->points.push_back(pointt);
	}

	Mat = Eigen::Matrix4f::Identity();
	int Iterations = _Iterations;
	int Rand_seed = Iterations;
	PointCloudPtr cloud_src_out, cloud_tar_out;
	float RANSAC_inlier_judge_thresh = Threshold_Radius * resolution;
	float size = 1.0f;
	if (cloud_source->size() < 5000)size = 1.0f;
	else if (cloud_source->size() < 12000)size = 2.0f;
	else if (cloud_source->size() < 18000)size = 2.5f;
	else if (cloud_source->size() < 25000)size = 3.0f;
	else if (cloud_source->size() < 32000)size = 3.5f;
	else size = 4.0f;
	cloud_src_out = downSampling(cloud_source, resolution, size);
	cloud_tar_out = downSampling(cloud_target, resolution, size);
	//float aver_distance = -9999;
	float aver_distance = 2e20;

	while (Iterations)
	{
		Rand_seed++;
		int Match_Idx1, Match_Idx2, Match_Idx3;
		bool flag = true;
		pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		while (flag)
		{
			Rand_seed++;//防止某次产生的伪随机数不符合要求，而随机种子不变，导致生成相同的伪随机数
			Rand_3(Rand_seed, Match.size(), Match_Idx1, Match_Idx2, Match_Idx3);
			point_s1 = cloud_source->points[Match[Match_Idx1].source_idx];
			point_s2 = cloud_source->points[Match[Match_Idx2].source_idx];
			point_s3 = cloud_source->points[Match[Match_Idx3].source_idx];
			point_t1 = cloud_target->points[Match[Match_Idx1].target_idx];
			point_t2 = cloud_target->points[Match[Match_Idx2].target_idx];
			point_t3 = cloud_target->points[Match[Match_Idx3].target_idx];
			float d1 = sqrt(pow(point_s1.x - point_s2.x, 2) + pow(point_s1.y - point_s2.y, 2) + pow(point_s1.z - point_s2.z, 2));
			float d2 = sqrt(pow(point_s1.x - point_s3.x, 2) + pow(point_s1.y - point_s3.y, 2) + pow(point_s1.z - point_s3.z, 2));
			float d3 = sqrt(pow(point_s2.x - point_s3.x, 2) + pow(point_s2.y - point_s3.y, 2) + pow(point_s2.z - point_s3.z, 2));
			if (d1 > Threshold_Radius * 2 * resolution && d2 > Threshold_Radius * 2 * resolution && d3 > Threshold_Radius * 2 * resolution)
			{
				flag = false;
				//cout << d1 / float(resolution) << " " << d2 / float(resolution) << " " << d3 / float(resolution) << endl;
			}
		}
		//
		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		float averDistance_iter = getAverDistance(cloud_src_out, cloud_tar_out, Mat_iter, resolution);
		//float averDistance_iter = Score_est(source_match_points, target_match_points, cloud_source, cloud_target, Mat_iter, RANSAC_inlier_judge_thresh, loss, resolution);
		if (averDistance_iter < aver_distance)
		{
			aver_distance = averDistance_iter;
			Mat = Mat_iter;
		}
		//if (aver_distance < 3.0*resolution)break;
		Iterations--;
	}
	return 1;
}

int TwoSAC_GC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ points, pointt;
		points = cloud_source->points[Match[i].source_idx];
		pointt = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(points);
		target_match_points->points.push_back(pointt);
	}

	Mat = Eigen::Matrix4f::Identity();
	int Iterations = _Iterations;
	float RANSAC_inlier_judge_thresh = threshold * resolution;
	int Rand_seed = Iterations;
	PointCloudPtr cloud_src_out, cloud_tar_out;
	float size = 1.0f;
	if (cloud_source->size() < 5000)size = 1.0f;
	else if (cloud_source->size() < 12000)size = 2.0f;
	else if (cloud_source->size() < 18000)size = 2.5f;
	else if (cloud_source->size() < 25000)size = 3.0f;
	else if (cloud_source->size() < 32000)size = 3.5f;
	else size = 4.0f;
	cloud_src_out = downSampling(cloud_source, resolution, size);
	cloud_tar_out = downSampling(cloud_target, resolution, size);
	float score = 0;
	while (Iterations)
	{
		Rand_seed++;
		int Match_Idx1, Match_Idx2;
		Rand_2(Rand_seed, Match.size(), Match_Idx1, Match_Idx2);
		pcl::PointXYZ cor_src1, cor_src2, cor_tar1, cor_tar2, point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		cor_src1 = cloud_source->points[Match[Match_Idx1].source_idx];
		cor_src2 = cloud_source->points[Match[Match_Idx2].source_idx];
		cor_tar1 = cloud_target->points[Match[Match_Idx1].target_idx];
		cor_tar2 = cloud_target->points[Match[Match_Idx2].target_idx];

		Vertex LRA_s1, LRA_s2, LRA_t1, LRA_t2;
		LRA_s1 = Match[Match_Idx1].source_LRF.z_axis;
		LRA_s2 = Match[Match_Idx2].source_LRF.z_axis;
		LRA_t1 = Match[Match_Idx1].target_LRF.z_axis;
		LRA_t2 = Match[Match_Idx2].target_LRF.z_axis;

		float delta_dis = Threshold_Radius * resolution, delta_angle = 15;
		float distance = fabs(sqrt(pow(cor_src1.x - cor_src2.x, 2) + pow(cor_src1.y - cor_src2.y, 2) + pow(cor_src1.z - cor_src2.z, 2))
			- sqrt(pow(cor_tar1.x - cor_tar2.x, 2) + pow(cor_tar1.y - cor_tar2.y, 2) + pow(cor_tar1.z - cor_tar2.z, 2)));
		float product1 = LRA_s1.x * LRA_s2.x + LRA_s1.y * LRA_s2.y + LRA_s1.z * LRA_s2.z;
		float dis1 = sqrt(pow(LRA_s1.x, 2 + pow(LRA_s1.y, 2) + pow(LRA_s1.z, 2))) * sqrt(pow(LRA_s2.x, 2 + pow(LRA_s2.y, 2) + pow(LRA_s2.z, 2)));
		float product2 = LRA_t1.x * LRA_t2.x + LRA_t1.y * LRA_t2.y + LRA_t1.z * LRA_t2.z;
		float dis2 = sqrt(pow(LRA_t1.x, 2 + pow(LRA_t1.y, 2) + pow(LRA_t1.z, 2))) * sqrt(pow(LRA_t2.x, 2 + pow(LRA_t2.y, 2) + pow(LRA_t2.z, 2)));
		float angle = fabs(acos(product1 / dis1) - acos(product2 / dis2));
		if (distance > delta_dis || angle > delta_angle)continue;

		//Iterations--;
		point_s1 = cor_src1;  point_s2 = cor_src2;
		point_s3.x = cor_src1.x + LRA_s1.x; point_s3.y = cor_src1.y + LRA_s1.y; point_s3.z = cor_src1.z + LRA_s1.z;
		point_t1 = cor_tar1; point_t2 = cor_tar2;
		point_t3.x = cor_tar1.x + LRA_t1.x; point_t3.y = cor_tar1.y + LRA_t1.y; point_t3.z = cor_tar1.z + LRA_t1.z;

		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		float alpha = Threshold_Radius * resolution;
		float score_iter = RANSAC_overlap(cloud_src_out, cloud_tar_out, Mat_iter, alpha, resolution);
		//float score_iter = Score_est(source_match_points, target_match_points, cloud_source, cloud_target, Mat_iter, RANSAC_inlier_judge_thresh, loss, resolution);
		if (score_iter > score)
		{
			score = score_iter;
			Mat = Mat_iter;
		}
		Iterations--;
	}
	return 1;
}
int CG_SAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, vector<Line>lines, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ points, pointt;
		points = cloud_source->points[Match[i].source_idx];
		pointt = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(points);
		target_match_points->points.push_back(pointt);
	}

	Mat = Eigen::Matrix4f::Identity();
	int Iterations = _Iterations;
	float RANSAC_inlier_judge_thresh = threshold * resolution;
	PointCloudPtr cloud_src_out, cloud_tar_out;
	float size = 1.0f;
	if (cloud_source->size() < 5000)size = 1.0f;
	else if (cloud_source->size() < 12000)size = 2.0f;
	else if (cloud_source->size() < 18000)size = 2.5f;
	else if (cloud_source->size() < 25000)size = 3.0f;
	else if (cloud_source->size() < 32000)size = 3.5f;
	else size = 4.0f;
	cloud_src_out = downSampling(cloud_source, resolution, size);
	cloud_tar_out = downSampling(cloud_target, resolution, size);
	float score = 0;
	for(int i = 0; i < Iterations; i++)
	{
		int Match_Idx1, Match_Idx2;
		Match_Idx1 = lines[i].index1;
		Match_Idx2 = lines[i].index2;
		pcl::PointXYZ cor_src1, cor_src2, cor_tar1, cor_tar2, point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		cor_src1 = cloud_source->points[Match[Match_Idx1].source_idx];
		cor_src2 = cloud_source->points[Match[Match_Idx2].source_idx];
		cor_tar1 = cloud_target->points[Match[Match_Idx1].target_idx];
		cor_tar2 = cloud_target->points[Match[Match_Idx2].target_idx];

		Vertex LRA_s1, LRA_s2, LRA_t1, LRA_t2;
		LRA_s1 = Match[Match_Idx1].source_LRF.z_axis;
		LRA_s2 = Match[Match_Idx2].source_LRF.z_axis;
		LRA_t1 = Match[Match_Idx1].target_LRF.z_axis;
		LRA_t2 = Match[Match_Idx2].target_LRF.z_axis;

		float delta_dis = Threshold_Radius * resolution, delta_angle = 15;
		float distance = fabs(sqrt(pow(cor_src1.x - cor_src2.x, 2) + pow(cor_src1.y - cor_src2.y, 2) + pow(cor_src1.z - cor_src2.z, 2))
			- sqrt(pow(cor_tar1.x - cor_tar2.x, 2) + pow(cor_tar1.y - cor_tar2.y, 2) + pow(cor_tar1.z - cor_tar2.z, 2)));
		float product1 = LRA_s1.x * LRA_s2.x + LRA_s1.y * LRA_s2.y + LRA_s1.z * LRA_s2.z;
		float dis1 = sqrt(pow(LRA_s1.x, 2 + pow(LRA_s1.y, 2) + pow(LRA_s1.z, 2))) * sqrt(pow(LRA_s2.x, 2 + pow(LRA_s2.y, 2) + pow(LRA_s2.z, 2)));
		float product2 = LRA_t1.x * LRA_t2.x + LRA_t1.y * LRA_t2.y + LRA_t1.z * LRA_t2.z;
		float dis2 = sqrt(pow(LRA_t1.x, 2 + pow(LRA_t1.y, 2) + pow(LRA_t1.z, 2))) * sqrt(pow(LRA_t2.x, 2 + pow(LRA_t2.y, 2) + pow(LRA_t2.z, 2)));
		float angle = fabs(acos(product1 / dis1) - acos(product2 / dis2));
		if (distance > delta_dis || angle > delta_angle)continue;

		//Iterations--;
		point_s1 = cor_src1;  point_s2 = cor_src2;
		point_s3.x = cor_src1.x + LRA_s1.x; point_s3.y = cor_src1.y + LRA_s1.y; point_s3.z = cor_src1.z + LRA_s1.z;
		point_t1 = cor_tar1; point_t2 = cor_tar2;
		point_t3.x = cor_tar1.x + LRA_t1.x; point_t3.y = cor_tar1.y + LRA_t1.y; point_t3.z = cor_tar1.z + LRA_t1.z;

		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		float alpha = Threshold_Radius * resolution;
		//float score_iter = RANSAC_overlap(cloud_src_out, cloud_tar_out, Mat_iter, alpha, resolution);
		float score_iter = Score_est(source_match_points, target_match_points, cloud_source, cloud_target, Mat_iter, RANSAC_inlier_judge_thresh, loss, resolution);
		if (score_iter > score)
		{
			score = score_iter;
			Mat = Mat_iter;
		}
	}
	return 1;
}
int GC1SAC(PointCloudPtr cloud_source, PointCloudPtr cloud_target, std::vector<Match_pair> Match,
	float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ points, pointt;
		points = cloud_source->points[Match[i].source_idx];
		pointt = cloud_target->points[Match[i].target_idx];
		source_match_points->points.push_back(points);
		target_match_points->points.push_back(pointt);
	}

	Mat = Eigen::Matrix4f::Identity();
	int Iterations = _Iterations;
	float RANSAC_inlier_judge_thresh = threshold * resolution;
	int Rand_seed = Iterations;
	PointCloudPtr cloud_src_out, cloud_tar_out;
	float size = 1.0f;
	if (cloud_source->size() < 5000)size = 1.0f;
	else if (cloud_source->size() < 12000)size = 2.0f;
	else if (cloud_source->size() < 18000)size = 2.5f;
	else if (cloud_source->size() < 25000)size = 3.0f;
	else if (cloud_source->size() < 32000)size = 3.5f;
	else size = 4.0f;
	cloud_src_out = downSampling(cloud_source, resolution, size);
	cloud_tar_out = downSampling(cloud_target, resolution, size);
	float score = 0;
	while (Iterations)
	{
		Rand_seed++;
		int Match_Idx;
		Rand_1(Rand_seed, Match.size(), Match_Idx);
		pcl::PointXYZ point_s, point_t, point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		point_s = cloud_source->points[Match[Match_Idx].source_idx];
		point_t = cloud_target->points[Match[Match_Idx].target_idx];
		LRF LRF_src = Match[Match_Idx].source_LRF;
		LRF LRF_tar = Match[Match_Idx].target_LRF;

		point_s1.x = LRF_src.x_axis.x + point_s.x; point_s1.y = LRF_src.x_axis.y + point_s.y; point_s1.z = LRF_src.x_axis.z + point_s.z;
		point_s2.x = LRF_src.y_axis.x + point_s.x; point_s2.y = LRF_src.y_axis.y + point_s.y; point_s2.z = LRF_src.y_axis.z + point_s.z;
		point_s3.x = LRF_src.z_axis.x + point_s.x; point_s3.y = LRF_src.z_axis.y + point_s.y; point_s3.z = LRF_src.z_axis.z + point_s.z;

		point_t1.x = LRF_tar.x_axis.x + point_t.x; point_t1.y = LRF_tar.x_axis.y + point_t.y; point_t1.z = LRF_tar.x_axis.z + point_t.z;
		point_t2.x = LRF_tar.y_axis.x + point_t.x; point_t2.y = LRF_tar.y_axis.y + point_t.y; point_t2.z = LRF_tar.y_axis.z + point_t.z;
		point_t3.x = LRF_tar.z_axis.x + point_t.x; point_t3.y = LRF_tar.z_axis.y + point_t.y; point_t3.z = LRF_tar.z_axis.z + point_t.z;
		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		float alpha = Threshold_Radius * resolution;
		float score_iter = RANSAC_overlap(cloud_src_out, cloud_tar_out, Mat_iter, alpha, resolution);
		//float score_iter = Score_est(source_match_points, target_match_points, cloud_source, cloud_target, Mat_iter, RANSAC_inlier_judge_thresh, loss, resolution);
		if (score_iter > score)
		{
			score = score_iter;
			Mat = Mat_iter;
		}
		Iterations--;
	}
	return 1;
}





float getAverDistance(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, Eigen::Matrix4f& Mat, float resolution)
{
	/*float averDistance = averDistanceInit;
	pcl::PointCloud<pcl::PointXYZ>::Ptr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_src, *src_trans, Mat);
	pcl::KdTreeFLANN<pcl::PointXYZ>tree;
	vector<int>indices;
	vector<float>dist;
	tree.setInputCloud(cloud_tar);
	float N = 0;
	float sumDis = 0.0f;
	float distance_threshold = Threshold_Radius * resolution;
	for (int i = 0; i < src_trans->size(); i++)
	{
		if (tree.nearestKSearch(src_trans->points[i], 1, indices, dist) > 0)
		{
			float sqdist = sqrt(dist[0]);
			if (sqdist < distance_threshold)
			{
				N++;
				sumDis += sqdist;
			}
		}
	}
	if (N / float(src_trans->size()) > 0.3)averDistance = sumDis / N;
*/
	float score = 0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_src, *src_trans, Mat);
	pcl::KdTreeFLANN<pcl::PointXYZ>tree;
	vector<int>indices;
	vector<float>dist;
	tree.setInputCloud(cloud_tar);
	float distance_threshold = Threshold_Radius * resolution, temp_score = 0;
	for (int i = 0; i < src_trans->size(); i++)
	{
		if (tree.nearestKSearch(src_trans->points[i], 1, indices, dist) > 0)
		{
			float sqdist = sqrt(dist[0]);
			if (sqdist < distance_threshold)
			{
				temp_score = (distance_threshold- sqdist) / distance_threshold;
				score += temp_score;
			}
		}
	}
	return score;

}
float HuberDistance(PointCloudPtr source_match_points, PointCloudPtr target_match_points, Eigen::Matrix4f& Mat, float resolution)
{
	float huber_dis_thresh = Threshold_Radius * resolution;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source_match_points, *source_match_points_trans, Mat);
	float sqdist = 0;
	float sumDis = 0.0f;
	for (int i = 0; i < source_match_points_trans->size(); i++)
	{
		sqdist = pow(source_match_points_trans->points[i].x - target_match_points->points[i].x, 2) +
			pow(source_match_points_trans->points[i].y - target_match_points->points[i].y, 2) +
			pow(source_match_points_trans->points[i].z - target_match_points->points[i].z, 2);
		if (sqrt(sqdist) < huber_dis_thresh)
		{
			sumDis += 0.5*sqdist;
		}
		else
		{
			sumDis += 0.5 * huber_dis_thresh * (2 * sqrt(sqdist) - huber_dis_thresh);
		}
	}
	return sumDis;
}
float RANSAC_overlap(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, Eigen::Matrix4f& Mat, float alpha, float resolution)
{
	int i;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_src, *cloud_trans, Mat);
	//
	int N = 0;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud_tar);
	std::vector<int> Idx;
	std::vector<float> Dist;
	//for (i = 0; i < cloud_trans->points.size(); i++)
	//{
	//	int judge = kdtree.radiusSearch(cloud_trans->points[i], alpha, Idx, Dist);
	//	if (judge > 0)
	//		N++;
	//}
	//return N;
	float score = 0, temp_score = 0;
	for (int i = 0; i < cloud_trans->size(); i++)
	{
		if (kdtree.nearestKSearch(cloud_trans->points[i], 1, Idx, Dist) > 0)
		{
			float sqdist = sqrt(Dist[0]);
			if (sqdist < alpha)
			{
				temp_score = (alpha - sqdist) / alpha;
				score += temp_score;
			}
		}
	}
	return score;
}