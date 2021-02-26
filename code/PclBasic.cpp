#include"header.h"

//read pointcloud
int XYZ_Read(string Filename, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	int i, nXYZ_nums;
	vector<Vertex> vXYZ;
	FILE* fp = fopen(Filename.c_str(), "r");
	if (fp == NULL)
	{
		printf("File can't open!\n");
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
	return 0;
}

//compute pointcloud resolution
double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices;
	std::vector<float> sqr_distances;
	pcl::search::KdTree<pcl::PointXYZ> tree;
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

//uniform dowm sampling for point cloud
PointCloudPtr downSampling(PointCloudPtr cloud_in, float resolution, float downSize)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> filter;
	filter.setInputCloud(cloud_in);
	filter.setLeafSize(downSize * resolution, downSize * resolution, downSize * resolution);
	filter.filter(*cloud_out);
	return cloud_out;
}

//get HarrisKeypoint3D
pcl::PointCloud<pcl::PointXYZ>::Ptr getHarrisKeypoint3D(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, float radius, float resolution)
{
	pcl::PointCloud<pcl::PointXYZI> result;
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr final_keyPoint(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI, pcl::Normal> harris;
	harris.setInputCloud(cloud_in);
	harris.setNonMaxSupression(true);
	harris.setRefine(false);
	harris.setRadius(radius);
	//设置参数过大，关键点数量为0，过小，关键点为数量为定值，可视化时会出现问题。
	//harris.setThreshold(0.8*resolution);
	harris.compute(result);
	pcl::copyPointCloud(result, *keyPoint);
	return keyPoint;
}

pcl::PointIndicesPtr getIndexofKeyPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint, std::vector<int>& keyPoint_Idx)
{
	pcl::PointIndicesPtr indices = boost::shared_ptr <pcl::PointIndices>(new pcl::PointIndices());
	//pcl::PointIndicesPtr indices = pcl::PointIndicesPtr(new pcl::PointIndices());
	//pcl::PointIndicesPtr indices(new pcl::PointIndices());
	std::vector<int>index;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<float> Dist;
	kdtree.setInputCloud(cloud_in);
	for (int i = 0; i < keyPoint->size(); i++)
	{
		kdtree.nearestKSearch(keyPoint->points[i], 1, index, Dist);
		indices->indices.push_back(index[0]);
		keyPoint_Idx.push_back(index[0]);
		//cout << Dist[0] << " ";
	}
	return indices;
}

PointCloudPtr removeInvalidkeyPoint(PointCloudPtr cloud_in, vector<int> &keyPointIdx, PointCloudPtr keyPoint, float resolution)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr final_keyPoint(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<int>index;
	std::vector<float> Dist;
	kdtree.setInputCloud(cloud_in);
	vector<int>keyPointTempIdx;
	for (int i = 0; i < keyPoint->size(); i++)
	{
		kdtree.radiusSearch(cloud_in->points[keyPointIdx[i]], 15 * resolution, index, Dist);
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

//compute SHOT descriptor
pcl::PointCloud<pcl::SHOT352>::Ptr SHOT_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, 
	vector<int>&indices, float sup_radius, vector<vector<float>>&features)
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
	shot_est.setRadiusSearch(sup_radius);
	shot_est.compute(*shots);
	features.resize(shots->points.size());
	for (i = 0; i < features.size(); i++)
	{
		features[i].resize(352);
		for (j = 0; j < 352; j++)
		{
			features[i][j] = shots->points[i].descriptor[j];
		}
	}
	return shots;
}

bool compair(const Match_pair& v1, const Match_pair& v2)
{
	return v1.ratio > v2.ratio;//降序排列
}
vector<Match_pair> getTopKCorresByRatio(pcl::PointCloud<pcl::SHOT352>::Ptr descriptors_src,
	pcl::PointCloud<pcl::SHOT352>::Ptr descriptors_tar, vector<int> keyPoint_index_src, vector<int> keyPoint_index_tar, int k)
{
	pcl::KdTreeFLANN<pcl::SHOT352>kdtree;
	vector<int>Idx;
	vector<float>Dist;
	kdtree.setInputCloud(descriptors_tar);
	std::vector<Match_pair> match_temp;
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
		match_temp.push_back(pair);
	}
	sort(match_temp.begin(), match_temp.end(), compair);
	for (int i = 0; i < k; i++)
	{
		match.push_back(match_temp[i]);
	}
	return match;
}

// 仿真生成指定正确率的匹配集
// k要生成匹配集的大小，correctRate正确率，resolution点云分辨率，threshold判断此匹配为正确匹配的阈值
vector<Match_pair> generateMatch(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target,
	 vector<int> keyPoint_index_src, vector<int> keyPoint_index_tar, Eigen::Matrix4f GT, int k, float correctRate, float resolution, float threshold)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_source, *cloud_source_trans, GT);

	//生成正确匹配的数量
	int correctNum = k * correctRate;

	std::vector<Match_pair> match;
	Match_pair pair;
	std::vector<int>index;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<float> Dist;
	kdtree.setInputCloud(cloud_target);

	for (int i = 0; i < keyPoint_index_src.size(); i++)
	{
		//生成的匹配数量已经足够
		if (match.size() >= k)break;
		//1.计算匹配源索引
		int indexSrc = keyPoint_index_src[i];
		//2.计算匹配目标索引
		int indexTar = -1;
		//仿真生成特定数量的正确匹配
		if (match.size() < correctNum)
		{
			kdtree.nearestKSearch(cloud_source_trans->points[indexSrc], 1, index, Dist);
			//确保距离小于特定阈值
			if (Dist[0] < threshold * resolution)
			{
				indexTar = index[0];
				index.clear();
				Dist.clear();
			}
			else continue;
		}
		//随机生成错误匹配
		else
		{
			Rand_1(i, cloud_target->size(), indexTar);
		}
		//加入匹配
		if (indexTar != -1)
		{
			pair.source_idx = indexSrc;
			pair.target_idx = indexTar;
			match.push_back(pair);
		}
	}
	return match;
}

float RMSE_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target,
	Eigen::Matrix4f&Mat_est, Eigen::Matrix4f&Mat_GT, float mr)
{
	float RMSE_temp = 0.0f;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_GT(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_source, *cloud_source_trans_GT, Mat_GT);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_EST(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_source, *cloud_source_trans_EST, Mat_est);
	vector<int>overlap_idx; float overlap_thresh = 4 * mr;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree1;
	pcl::PointXYZ query_point;
	vector<int>pointIdx;
	vector<float>pointDst;
	kdtree1.setInputCloud(cloud_target);
	for (int i = 0; i < cloud_source_trans_GT->points.size(); i++)
	{
		query_point = cloud_source_trans_GT->points[i];
		kdtree1.nearestKSearch(query_point, 1, pointIdx, pointDst);
		if (sqrt(pointDst[0]) <= overlap_thresh)
			overlap_idx.push_back(i);
	}
	//
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree2;
	kdtree2.setInputCloud(cloud_source_trans_GT);
	for (int i = 0; i < overlap_idx.size(); i++)
	{
		//query_point = cloud_source_trans_EST->points[overlap_idx[i]];
		//kdtree2.nearestKSearch(query_point,1,pointIdx,pointDst);	RMSE_temp+=sqrt(pointDst[0]);
		float dist_x = pow(cloud_source_trans_EST->points[overlap_idx[i]].x - cloud_source_trans_GT->points[overlap_idx[i]].x, 2);
		float dist_y = pow(cloud_source_trans_EST->points[overlap_idx[i]].y - cloud_source_trans_GT->points[overlap_idx[i]].y, 2);
		float dist_z = pow(cloud_source_trans_EST->points[overlap_idx[i]].z - cloud_source_trans_GT->points[overlap_idx[i]].z, 2);
		float dist = sqrt(dist_x + dist_y + dist_z);
		RMSE_temp += dist;
	}
	RMSE_temp /= overlap_idx.size();
	RMSE_temp /= mr;
	//
	return RMSE_temp;
}
vector<Match_pair> getTopKCorresByRatio(pcl::PointCloud<pcl::SHOT352>::Ptr descriptors_src,
	pcl::PointCloud<pcl::SHOT352>::Ptr descriptors_tar, vector<int> keyPoint_index_src, vector<int> keyPoint_index_tar, float ratio)
{
	pcl::KdTreeFLANN<pcl::SHOT352>kdtree;
	vector<int>Idx;
	vector<float>Dist;
	kdtree.setInputCloud(descriptors_tar);
	std::vector<Match_pair> match_temp;
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
		match_temp.push_back(pair);
	}
	sort(match_temp.begin(), match_temp.end(), compair);
	int k = match_temp.size()*ratio;
	for (int i = 0; i < k; i++)
	{
		match.push_back(match_temp[i]);
	}
	return match;
}

vector<Match_pair> getTopKCorresByRatio1344(pcl::PointCloud<pcl::SHOT1344>::Ptr descriptors_src,
	pcl::PointCloud<pcl::SHOT1344>::Ptr descriptors_tar, vector<int> keyPoint_index_src, vector<int> keyPoint_index_tar, float ratio)
{
	pcl::KdTreeFLANN<pcl::SHOT1344>kdtree;
	vector<int>Idx;
	vector<float>Dist;
	cout << descriptors_tar->size() << endl;
	kdtree.setInputCloud(descriptors_tar);
	std::vector<Match_pair> match_temp;
	std::vector<Match_pair> match;
	Match_pair pair;
	pcl::SHOT1344 n1, n2, src_i;
	for (int i = 0; i < descriptors_src->size(); i++)
	{
		kdtree.nearestKSearch(descriptors_src->points[i], 2, Idx, Dist);
		float up = sqrt(Dist[0]), down = sqrt(Dist[1]);
		pair.source_idx = keyPoint_index_src[i];
		pair.target_idx = keyPoint_index_tar[Idx[0]];
		if (down < 1e-6)continue;
		pair.ratio = 1.0 - (up / down);
		match_temp.push_back(pair);
	}
	sort(match_temp.begin(), match_temp.end(), compair);
	int k = match_temp.size()*ratio;
	for (int i = 0; i < k; i++)
	{
		match.push_back(match_temp[i]);
	}
	return match;
}

int getCorrectCorrespondences(vector<Match_pair> match, vector<int>&corrcet_match, Eigen::Matrix4f Mat, float thresh,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Src(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Tar(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloud_Trans(new pcl::PointCloud <pcl::PointXYZ>);
	pcl::PointXYZ Point_Temp_s, Point_Temp_t;
	for (int i = 0; i < match.size(); i++)
	{
		Point_Temp_s = cloud_src->points[match[i].source_idx];
		Point_Temp_t = cloud_tar->points[match[i].target_idx];
		keyPointCloud_Src->push_back(Point_Temp_s);
		keyPointCloud_Tar->push_back(Point_Temp_t);
	}
	pcl::transformPointCloud(*keyPointCloud_Src, *keyPointCloud_Trans, Mat);
	int num = 0;
	for (int i = 0; i < match.size(); i++)
	{
		float x1 = keyPointCloud_Trans->points[i].x;
		float y1 = keyPointCloud_Trans->points[i].y;
		float z1 = keyPointCloud_Trans->points[i].z;
		float x2 = keyPointCloud_Tar->points[i].x;
		float y2 = keyPointCloud_Tar->points[i].y;
		float z2 = keyPointCloud_Tar->points[i].z;
		float dis = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
		if (dis <= thresh)
		{
			num++;
			corrcet_match.push_back(i);
		}
	}
	return num;
}

void Add_Gaussian_noise(float dev, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_noise)
{
	boost::mt19937 rng;
	rng.seed(static_cast<unsigned int> (1));
	boost::normal_distribution<> nd(0, dev);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
	cloud_noise->points.resize(cloud->points.size());
	cloud_noise->header = cloud->header;
	cloud_noise->width = cloud->width;
	cloud_noise->height = cloud->height;

	for (size_t point_i = 0; point_i < cloud->points.size(); ++point_i)
	{
		cloud_noise->points[point_i].x = cloud->points[point_i].x + static_cast<float> (var_nor());
		cloud_noise->points[point_i].y = cloud->points[point_i].y + static_cast<float> (var_nor());
		cloud_noise->points[point_i].z = cloud->points[point_i].z + static_cast<float> (var_nor());
	}
}
void GUO_ICP(PointCloudPtr&cloud_source, PointCloudPtr&cloud_target, float&mr, int&Max_iter_Num, Eigen::Matrix4f&Mat_ICP)
{
	int number_of_sample_points;
	float residual_error = 4.0f;//为mr的倍数，初始选max/20点
	float inlier_thresh = 4.0f;
	Mat_ICP = Eigen::Matrix4f::Identity();
	//int n_min=1000,n_max=cloud_source->points.size();
	//if(n_max<=n_min) n_min=n_max;//防止source中点过少

	for (int i = 0; i < Max_iter_Num; i++)
	{
		//printf("ICP Iter.%d ,",i+1);
		//source点云变换
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*cloud_source, *cloud_source_trans, Mat_ICP);
		//随机采样
		number_of_sample_points = cloud_source_trans->points.size() / pow(3.0f, residual_error);
		//printf("No. of Sample points:%d ,",number_of_sample_points);
		vector<int> Sample_cloud_Idx;
		boost::uniform_int<> distribution(0, cloud_source_trans->points.size());
		boost::mt19937 engine;
		boost::variate_generator<boost::mt19937, boost::uniform_int<> > myrandom(engine, distribution);
		for (int j = 0; j < number_of_sample_points; j++)
			Sample_cloud_Idx.push_back(myrandom());
		//Rand(cloud_source_trans->points.size()-1,number_of_sample_points,Sample_cloud_Idx);//注意-1，随机数生成器0~max
		Eigen::Matrix4f Mat_i;
		//利用最近点对估计变换矩阵并计算变换误差
		int flag = Iter_trans_est(cloud_source_trans, cloud_target, mr, inlier_thresh, Sample_cloud_Idx, residual_error, Mat_i);
		if (flag == -1)
		{
			printf("阈值过小，无法找到匹配点！\n");
			break;
		}
		//printf("Trans error (mr):%f\n",residual_error);
		//
		//inlier_thresh=1.8*residual_error;
		//inlier_thresh=1.8*residual_error;
		//inlier_thresh=exp(residual_error/mr-1);
		//inlier_thresh=log(residual_error/mr+1)*e_max;
		//Mat_ICP*=Mat_i;
		Mat_ICP = Mat_i * Mat_ICP;
		if (residual_error <= 0.01) break;
	}
}
int Iter_trans_est(PointCloudPtr&cloud_source, PointCloudPtr&cloud_target, float&mr, float&inlier_thresh, vector<int>&Sample_cloud_Idx, float&residual_error, Eigen::Matrix4f&Mat)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr closet_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr closet_target(new pcl::PointCloud<pcl::PointXYZ>);
	//
	residual_error = 0;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int> Idx;
	vector<float> Dist;
	kdtree.setInputCloud(cloud_target);
	for (int i = 0; i < Sample_cloud_Idx.size(); i++)
	{
		kdtree.nearestKSearch(cloud_source->points[Sample_cloud_Idx[i]], 1, Idx, Dist);
		if (sqrt(Dist[0]) <= inlier_thresh * mr)//剔除外点
		{
			closet_source->points.push_back(cloud_source->points[Sample_cloud_Idx[i]]);
			closet_target->points.push_back(cloud_target->points[Idx[0]]);
			residual_error += sqrt(Dist[0]);
		}
	}
	if (closet_source->points.size() == 0)
		return -1;
	else
	{
		residual_error /= closet_source->points.size();
		residual_error /= mr;
		//估计变换矩阵
		pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> SVD;
		SVD.estimateRigidTransformation(*closet_source, *closet_target, Mat);
	}
	return 0;
}

pcl::PointCloud<pcl::Histogram<153> >::Ptr SpinImage_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
	vector<int>&indices, float sup_radius, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
	int i, j;
	pcl::PointIndicesPtr Idx = boost::shared_ptr <pcl::PointIndices>(new pcl::PointIndices());
	for (j = 0; j < indices.size(); j++)
		Idx->indices.push_back(indices[j]);

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153>> spin_image_est;
	spin_image_est.setInputCloud(cloud);
	spin_image_est.setInputNormals(normals);
	pcl::PointCloud<pcl::Histogram<153> >::Ptr spin_images(new pcl::PointCloud<pcl::Histogram<153> >);
	spin_image_est.setSearchMethod(tree);
	spin_image_est.setIndices(Idx);
	spin_image_est.setRadiusSearch(sup_radius);
	spin_image_est.compute(*spin_images);
	return spin_images;
}
void pointLFSH(float r, int bin_num, Vertex &searchPoint, Vertex& n, pcl::PointCloud<pcl::PointXYZ>::Ptr&neigh, pcl::PointCloud<pcl::Normal>::Ptr&sphere_normals,
	vector<float> &histogram)
{
	int i;
	float nx = n.x;
	float ny = n.y;
	float nz = n.z;
	float x0 = searchPoint.x - nx * r;//x0,y0,z0为searchpoint在平面上的投影
	float y0 = searchPoint.y - ny * r;
	float z0 = searchPoint.z - nz * r;
	float plane_D = -(nx*x0 + ny * y0 + nz * z0);
	//int depth_bin_num=bin_num/3;//参数可调
	int depth_bin_num = bin_num / 3;
	float depth_stride = 2 * r / depth_bin_num;
	int depth_bin_id;
	vector<float> depth_histogram(depth_bin_num, 0);
	/***************/
	//int density_bin_num=bin_num/3;
	int density_bin_num = bin_num / 6;
	float density_stride = r / density_bin_num;
	int density_bin_id;
	vector<float> density_histogram(density_bin_num, 0);
	/***************/
	//int angle_bin_num=bin_num/3;
	int angle_bin_num = bin_num / 2;
	float angle_stride = 180.0f / angle_bin_num;
	int angle_bin_id;
	vector<float> angle_histogram(angle_bin_num, 0);
	/***************/
	float a, b, c;
	float temp_depth, temp_radius, temp_angle;
	for (i = 0; i < neigh->points.size(); i++)
	{
		temp_depth = nx * neigh->points[i].x + ny * neigh->points[i].y + nz * neigh->points[i].z + plane_D;
		c = (neigh->points[i].x - searchPoint.x)*(neigh->points[i].x - searchPoint.x) +
			(neigh->points[i].y - searchPoint.y)*(neigh->points[i].y - searchPoint.y) +
			(neigh->points[i].z - searchPoint.z)*(neigh->points[i].z - searchPoint.z);
		b = (neigh->points[i].x - searchPoint.x)*nx + (neigh->points[i].y - searchPoint.y)*ny + (neigh->points[i].z - searchPoint.z)*nz;
		a = sqrt(abs(c - b * b));
		temp_radius = a;
		temp_angle = sphere_normals->points[i].normal_x*nx + sphere_normals->points[i].normal_y*ny + sphere_normals->points[i].normal_z*nz;
		if (temp_angle > 1)
			temp_angle = 1;
		if (temp_angle < -1)
			temp_angle = -1;
		temp_angle = acos(temp_angle) / M_PI * 180;
		//统计直方图
		if (temp_depth >= 2 * r)//防止浮点数溢出
			depth_bin_id = depth_bin_num;
		if (temp_depth <= 0.0f)
			depth_bin_id = 1;
		else
			depth_bin_id = temp_depth / depth_stride + 1;

		if (temp_radius >= r)
			density_bin_id = density_bin_num;
		else
			density_bin_id = temp_radius / density_stride + 1;

		if (temp_angle >= 180)
			angle_bin_id = angle_bin_num;
		else
			angle_bin_id = temp_angle / angle_stride + 1;
		//
		depth_histogram[depth_bin_id - 1] += 1 / float(neigh->points.size());
		density_histogram[density_bin_id - 1] += 1 / float(neigh->points.size());
		angle_histogram[angle_bin_id - 1] += 1 / float(neigh->points.size());
	}
	copy(density_histogram.begin(), density_histogram.end(), back_inserter(depth_histogram));
	copy(angle_histogram.begin(), angle_histogram.end(), back_inserter(depth_histogram));
	histogram = depth_histogram;
}
//sup_radius:15*pr;bin_num:30;
void LFSH_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<int> indices, float sup_radius, int bin_num, vector<vector<float>>&Histograms)
{
	int i, j;
	///////////////////////计算法向量/////////////////////////////
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);
	///////////////////计算包围球//////////////////////////////
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int>pointIdx;
	vector<float>pointDst;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZ query_point;
	Vertex query_p, query_normal;
	//
	for (i = 0; i < indices.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::Normal>::Ptr sphere_normals(new pcl::PointCloud<pcl::Normal>);
		vector<float>hist_temp;
		query_point = cloud->points[indices[i]];
		if (kdtree.radiusSearch(query_point, sup_radius, pointIdx, pointDst) > 5)
		{
			for (j = 0; j < pointIdx.size(); j++)
			{
				sphere_neighbor->points.push_back(cloud->points[pointIdx[j]]);
				sphere_normals->points.push_back(normals->points[pointIdx[j]]);
			}
			Vertex LRA = { normals->points[indices[i]].normal_x,normals->points[indices[i]].normal_y,normals->points[indices[i]].normal_z };
			query_p.x = query_point.x;
			query_p.y = query_point.y;
			query_p.z = query_point.z;
			query_normal.x = normals->points[indices[i]].normal_x;
			query_normal.y = normals->points[indices[i]].normal_y;
			query_normal.z = normals->points[indices[i]].normal_z;
			pointLFSH(sup_radius, bin_num, query_p, query_normal, sphere_neighbor, sphere_normals, hist_temp);
			//
			Histograms.push_back(hist_temp);
		}
		else
		{
			vector<float> f_null(bin_num, 0.0f);
			Histograms.push_back(f_null);
		}
	}
}
pcl::PointCloud<pcl::SHOT352>::Ptr SpinImageToShot(pcl::PointCloud<pcl::Histogram<153> >::Ptr SpinImage)
{
	pcl::PointCloud<pcl::SHOT352>::Ptr shot(new pcl::PointCloud<pcl::SHOT352>);
	for (int i = 0; i < SpinImage->size(); i++)
	{
		pcl::SHOT352 shotTemp;
		for (int j = 0; j < 352; j++)
		{
			if (j < 153)
			{
				shotTemp.descriptor[j] = SpinImage->points[i].histogram[j];
			}
			else shotTemp.descriptor[j] = 0;
		}
		shot->push_back(shotTemp);
	}
	return shot;
}
pcl::PointCloud<pcl::SHOT352>::Ptr LFSHToShot(vector<vector<float>>LFSH)
{
	pcl::PointCloud<pcl::SHOT352>::Ptr shot(new pcl::PointCloud<pcl::SHOT352>);
	pcl::SHOT352 shotTemp;
	for (int i = 0; i < LFSH.size(); i++)
	{
		for (int j = 0; j < 352; j++)
		{
			if (j < LFSH[i].size())
			{
				shotTemp.descriptor[j] = LFSH[i][j];
			}
			else shotTemp.descriptor[j] = 0;
		}
		shot->push_back(shotTemp);
	}
	return shot;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr getISS3dKeypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, float resolution)
{
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_det;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint(new pcl::PointCloud<pcl::PointXYZ>);
	//参数设置
	iss_det.setSearchMethod(tree);
	iss_det.setSalientRadius(2.7 * resolution);//
	iss_det.setNonMaxRadius(1.8 * resolution);//
	iss_det.setThreshold21(0.975);
	iss_det.setThreshold32(0.975);
	iss_det.setMinNeighbors(5);
	iss_det.setNumberOfThreads(4);
	iss_det.setInputCloud(cloud_in);
	iss_det.compute(*keyPoint);
	return keyPoint;
}
pcl::PointCloud<pcl::SHOT1344>::Ptr VOIDToShot(vector<vector<float>>Three_Dist)
{
	pcl::PointCloud<pcl::SHOT1344>::Ptr shot(new pcl::PointCloud<pcl::SHOT1344>);
	pcl::SHOT1344 shotTemp;
	for (int i = 0; i < Three_Dist.size(); i++)
	{
		for (int j = 0; j < 1344; j++)
		{
			if (j < Three_Dist[i].size())
			{
				shotTemp.descriptor[j] = Three_Dist[i][j];
			}
			else shotTemp.descriptor[j] = 0;
		}
		shot->push_back(shotTemp);
	}
	return shot;
}
int Three_Dist_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<int>&indices, std::vector<int>&Valid_Idx, float&sup_radius, std::vector<std::vector<float>>&features)
{
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n1;
	pcl::PointCloud<pcl::Normal>::Ptr normals1(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr treeI(new pcl::search::KdTree<pcl::PointXYZ>);
	treeI->setInputCloud(cloud);
	n1.setInputCloud(cloud);
	n1.setSearchMethod(treeI);
	n1.setRadiusSearch(sup_radius / 4);
	n1.compute(*normals1);

	int map_cnt[10][10][10];
	float cell_width = sup_radius / 10;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<int>pointIdx;
	std::vector<float>pointDst;
	std::vector<float>feature_temp;
	kdtree.setInputCloud(cloud);
	//std::cout << sup_radius <<"  "<<cell_width << std::endl;
	for (int i = 0; i < indices.size(); i++)
	{


		pcl::PointXYZ query_point;
		query_point = cloud->points[indices[i]];



		pointIdx.clear();
		pointDst.clear();
		feature_temp.clear();
		Vertex pq, np, nq;
		np.x = normals1->points.at(indices[i]).normal_x;
		np.y = normals1->points.at(indices[i]).normal_y;
		np.z = normals1->points.at(indices[i]).normal_z;
		if (!pcl_isfinite(np.x) || !pcl_isfinite(np.y) || !pcl_isfinite(np.z)) {
			continue;
		}
		if (kdtree.radiusSearch(query_point, sup_radius, pointIdx, pointDst) > 10) {
			Valid_Idx.push_back(indices[i]);


			int sum = 0;
			for (int j = 0; j < pointIdx.size(); j++)
			{
				pq.x = cloud->points[pointIdx[j]].x - cloud->points[indices[i]].x;
				pq.y = cloud->points[pointIdx[j]].y - cloud->points[indices[i]].y;
				pq.z = cloud->points[pointIdx[j]].z - cloud->points[indices[i]].z;

				nq.x = normals1->points.at(pointIdx[j]).normal_x;
				nq.y = normals1->points.at(pointIdx[j]).normal_y;
				nq.z = normals1->points.at(pointIdx[j]).normal_z;

				if (!pcl_isfinite(nq.x) || !pcl_isfinite(nq.y) || !pcl_isfinite(nq.z)) {
					continue;
				}


				float dist_pq = sqrt(fabs(pq.x * pq.x + pq.y * pq.y + pq.z * pq.z));
				float dist_np_q = fabs(np.x * pq.x + np.y * pq.y + np.z * pq.z);
				pq.x *= -1;
				pq.y *= -1;
				pq.z *= -1;
				float dist_nq_p = fabs(nq.x * pq.x + nq.y * pq.y + nq.z *  pq.z);

				int index_i = int(dist_pq / cell_width), index_j = int(dist_np_q / cell_width), index_k = int(dist_nq_p / cell_width);
				if (index_i < 0 || index_j < 0 || index_k < 0 || index_i > 9 || index_j > 9 || index_k > 9) {
					continue;
				}
				sum++;

				map_cnt[index_i][index_j][index_k]++;

			}

			for (int k = 0; k < 10; k++)
			{
				for (int r = 0; r < 10; r++) {
					for (int q = 0; q < 10; q++) {
						feature_temp.push_back(float(map_cnt[k][r][q] * 1.0) / sum);
						map_cnt[k][r][q] = 0;
					}
				}
			}
			features.push_back(feature_temp);
		}
	}
	return 0;
}
//求三角形面积;
//返回-1为不能组成三角形;
double count_triangle_area(pcl::PointXYZ a, pcl::PointXYZ b, pcl::PointXYZ c) {
	double area = -1;

	double side[3];//存储三条边的长度;

	side[0] = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
	side[1] = sqrt(pow(a.x - c.x, 2) + pow(a.y - c.y, 2) + pow(a.z - c.z, 2));
	side[2] = sqrt(pow(c.x - b.x, 2) + pow(c.y - b.y, 2) + pow(c.z - b.z, 2));

	//不能构成三角形;
	if (side[0] + side[1] <= side[2] || side[0] + side[2] <= side[1] || side[1] + side[2] <= side[0]) return area;

	//利用海伦公式。s=sqr(p*(p-a)(p-b)(p-c)); 
	double p = (side[0] + side[1] + side[2]) / 2; //半周长;
	area = sqrt(p*(p - side[0])*(p - side[1])*(p - side[2]));

	return area;
}
float getOverlapRate(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_i, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_j, Eigen::Matrix4f GroundTruth, float resolution)
{
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<float> Dist;
	std::vector<int>index;
	kdtree.setInputCloud(cloud_j);
	int cloudSize = cloud_i->size();
	int inliner = 0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_i, *cloud_trans_src, GroundTruth);
	for (int i = 0; i < cloudSize; i++)
	{
		kdtree.nearestKSearch(cloud_trans_src->points[i], 1, index, Dist);
		if (Dist[0] < 10 * resolution)inliner++;
	}
	return (float)inliner / (float)cloudSize;
}
bool compairCircleByDgree(const Circle& c1, const Circle& c2)
{
	return c1.degree > c2.degree;//降序排列
}
bool compairCircleByCompatibilityDist(const Circle& c1, const Circle& c2)
{
	return c1.compatibility_dist > c2.compatibility_dist;//降序排列
}
bool compairCircleByArea(const Circle& c1, const Circle& c2)
{
	return c1.area > c2.area;//降序排列
}
bool compairNodeBydegree(const Node& n1, const Node& n2)
{
	return n1.degree > n2.degree;
}
bool compairTripleBydegree(const Triple& t1, const Triple& t2)
{
	return t1.degree > t2.degree;
}
bool compairLineByCompatibilityDist(const Line& l1, const Line& l2)
{
	return l1.compatibility_dist > l2.compatibility_dist;
}