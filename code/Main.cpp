#include"header.h"
bool compairCircleByDgree(const Circle& c1, const Circle& c2)
{
	return c1.degree > c2.degree;
}
bool compairCircleByCompatibilityDist(const Circle& c1, const Circle& c2)
{
	return c1.compatibility_dist > c2.compatibility_dist;
}
bool compairLineByCompatibilityDist(const Line& l1, const Line& l2)
{
	return l1.compatibility_dist > l2.compatibility_dist;
}
bool compairNodeBydegree(const Node& n1, const Node& n2)
{
	return n1.degree > n2.degree;
}
bool compairTripleBydegree(const Triple& t1, const Triple& t2)
{
	return t1.degree > t2.degree;
}
int main(int argc, char **argv)
{
	ifstream infile("gt.txt");
	
	Eigen::Matrix4f GT;

	//read GT metric
	infile >> GT(0, 0) >> GT(0, 1) >> GT(0, 2) >> GT(0, 3) >>
		GT(1, 0) >> GT(1, 1) >> GT(1, 2) >> GT(1, 3) >>
		GT(2, 0) >> GT(2, 1) >> GT(2, 2) >> GT(2, 3) >>
		GT(3, 0) >> GT(3, 1) >> GT(3, 2) >> GT(3, 3);

	//read pointcloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar(new pcl::PointCloud <pcl::PointXYZ>);
	int a = XYZ_Read("chef_view1.txt", cloud_src);
	int b = XYZ_Read("chef_view2.txt", cloud_tar);


	//compute resolution
	float res_src = static_cast<float> (computeCloudResolution(cloud_src));
	float res_tar = static_cast<float> (computeCloudResolution(cloud_tar));
	float resolution = (res_src + res_tar) / 2;
	cout << "Before downsampling resolution=" << resolution << endl;
	float size = 2;
	cout << "Before downsampling cloud_src->size()=" << cloud_src->size() << endl;
	cout << "Befter downsampling cloud_tar->size()=" << cloud_tar->size() << endl;

	//downsampling
	cloud_src = downSampling(cloud_src, resolution, size);
	cloud_tar = downSampling(cloud_tar, resolution, size);
	cout << "After downsampling cloud_src->size()=" << cloud_src->size() << endl;
	cout << "After downsampling cloud_tar->size()=" << cloud_tar->size() << endl;

	res_src = static_cast<float> (computeCloudResolution(cloud_src));
	res_tar = static_cast<float> (computeCloudResolution(cloud_tar));
	resolution = (res_src + res_tar) / 2;
	cout << "After downsampling resolution=" << resolution << endl;
	float haris3D_radius = resolution * 2.0;//3.0 ~1.5k keypnts


	//compute normals
	pcl::PointCloud<pcl::Normal>::Ptr normals_src(new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::Normal>::Ptr normals_tar(new pcl::PointCloud<pcl::Normal>());
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
	norm_est.setKSearch(50);
	norm_est.setInputCloud(cloud_src);
	norm_est.compute(*normals_src);
	norm_est.setInputCloud(cloud_tar);
	norm_est.compute(*normals_tar);


	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint_src, keyPoint_tar;
	std::vector<int> keyPoint_index_src, keyPoint_index_tar;
	pcl::PointIndicesPtr temp_index_src, temp_index_tar;
	vector<vector<float>>features_src, features_tar;
	pcl::PointCloud<pcl::SHOT352>::Ptr descriptors_src, descriptors_tar;
	//compute keyPoints
	keyPoint_src = getHarrisKeypoint3D(cloud_src, haris3D_radius, resolution);
	keyPoint_tar = getHarrisKeypoint3D(cloud_tar, haris3D_radius, resolution);
	std::cout << "keyPoint_src" << keyPoint_src->size() << std::endl;
	std::cout << "keyPoint_tar" << keyPoint_tar->size() << std::endl;
	temp_index_src = getIndexofKeyPoint(cloud_src, keyPoint_src, keyPoint_index_src);
	temp_index_tar = getIndexofKeyPoint(cloud_tar, keyPoint_tar, keyPoint_index_tar);
	keyPoint_src = removeInvalidkeyPoint(cloud_src, keyPoint_index_src, keyPoint_src, resolution);
	keyPoint_tar = removeInvalidkeyPoint(cloud_tar, keyPoint_index_tar, keyPoint_tar, resolution);
	//compute descriptors
	descriptors_src = SHOT_compute(cloud_src, keyPoint_index_src, 2 * Threshold_Radius * resolution, features_src);
	descriptors_tar = SHOT_compute(cloud_tar, keyPoint_index_tar, 2 * Threshold_Radius * resolution, features_tar);

	//estimate correspondences
	std::vector<Match_pair> match;
	match = getTopKCorresByRatio(descriptors_src, descriptors_tar, keyPoint_index_src, keyPoint_index_tar, 100);
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < match.size(); i++)
	{
		int source_idx = match[i].source_idx;
		int target_idx = match[i].target_idx;
		source_match_points->points.push_back(cloud_src->points[source_idx]);
		target_match_points->points.push_back(cloud_tar->points[target_idx]);
	}

	vector<vector<double>>compatibility_matrix(100, vector<double>(100, 0));
	double c_dist_num[100];
	vector<Node> nodes;
	vector<vector<int>>adjacent_matrix(100, vector<int>(100, 0));

	//Initializing nodes
	for (int i = 0; i < match.size(); i++)
	{
		Node node;
		node.degree = 0;
		node.index = i;
		nodes.push_back(node);
	}
	computeMatiax(cloud_src, cloud_tar, match, normals_src, normals_tar, 0.9, nodes, adjacent_matrix, compatibility_matrix, resolution);
	vector<Circle> circles;
	vector<Line> lines;
	circles = getCircle(adjacent_matrix, 100, nodes, compatibility_matrix);
	lines = getLine(adjacent_matrix, nodes, compatibility_matrix, 100);
	sort(circles.begin(), circles.end(), compairCircleByCompatibilityDist);
	sort(lines.begin(), lines.end(), compairLineByCompatibilityDist);
	//Sort the nodes by degrees
	sort(nodes.begin(), nodes.end(), compairNodeBydegree);
	//Select the node of Top30 to form triples
	int topK = 30;
	vector<Triple>triples = getTriple(nodes, topK);
	sort(triples.begin(), triples.end(), compairTripleBydegree);
	Eigen::Matrix4f Mat;
	float RMSE;
	int m;

	int iteratorNums = 200;
	//RANSAC function entrance, Mat for SAC-COT algorithm output
	m = GuideSampling_score(cloud_src, cloud_tar, circles, triples, match, resolution, iteratorNums, 7.5, Mat, "Inlier");
	//Compare the difference between Mat and GT, and measure the algorithm
	RMSE = RMSE_compute(cloud_src, cloud_tar, Mat, GT, resolution);
	return 0;
}
