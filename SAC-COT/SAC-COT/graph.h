#pragma once
#include"geometry.h"
#include"CApp.h"
namespace sac_cot
{

	class Graph
	{
	public:
		Graph() {};
		void computeMatiax(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>pointcloud, std::vector<Match_pair>match,
			std::vector<pcl::PointCloud<pcl::Normal>::Ptr>normals, double threshod, double resolution);
		void nodesInitialization(std::vector<Match_pair> match);
		void computeCircle();
		void computeLine();
		void computeTriple(int topK = 30);
		std::vector<Circle>getCircles();
		std::vector<Triple>getTriples();

	private:
		std::vector<std::vector<int>>_adjacent_matrix;
		std::vector<std::vector<double>>_compatibility_matrix;
		std::vector<Node> _nodes;
		std::vector<Circle> _circles;
		std::vector<Triple>_triples;
		std::vector<Line> _lines;
	};

	double getCompatibility(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j,
		pcl::Normal &ns_i, pcl::Normal &ns_j, pcl::Normal &nt_i, pcl::Normal &nt_j, float resolution);
	double NormalDistance(pcl::Normal &n_i, pcl::Normal &n_j);
	float Rigidity(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j);
	double Distance(pcl::PointXYZ &A, pcl::PointXYZ &B);
	double Square(float x);
}
