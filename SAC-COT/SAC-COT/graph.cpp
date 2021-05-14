#include"graph.h"
using namespace sac_cot;

void Graph::nodesInitialization(std::vector<Match_pair> match)
{
	for (int i = 0; i < match.size(); i++)
	{
		Node node;
		node.degree = 0;
		node.index = i;
		_nodes.push_back(node);
	}
	sort(_nodes.begin(), _nodes.end(), [](Node n1, Node n2) {return n1.degree > n2.degree; });
}
void Graph::computeMatiax(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>pointcloud, std::vector<Match_pair>match, 
	std::vector<pcl::PointCloud<pcl::Normal>::Ptr>normals, double threshod, double resolution)
{
	_adjacent_matrix = std::vector<std::vector<int>>(_nodes.size(), std::vector<int>(_nodes.size(), 0));
	_compatibility_matrix = std::vector<std::vector<double>>(_nodes.size(), std::vector<double>(_nodes.size(), 0));

	int src_idxi, tar_idxi, src_idxj, tar_idxj;
	pcl::Normal src_normali, tar_normali, src_normalj, tar_normalj;
	pcl::PointXYZ sp_i, sp_j, tp_i, tp_j;
	double compatibility_dist;

	auto cloud_src = pointcloud[0], cloud_tar = pointcloud[1];
	auto normals_src = normals[0], normals_tar = normals[1];

	for (int i = 0; i < match.size(); i++)//选择第一根匹配
	{
		src_idxi = match[i].source_idx;
		tar_idxi = match[i].target_idx;//获取匹配对应的源点云与目标点云的索引
		sp_i = cloud_src->points[src_idxi];
		tp_i = cloud_tar->points[tar_idxi];//获取对应点
		src_normali = normals_src->points[src_idxi];
		tar_normali = normals_tar->points[tar_idxi];//获取法线
		for (int j = 0; j < match.size(); j++)//选择第二根匹配
		{
			if (i != j)//两根匹配不相同时进行下一步计算
			{
				src_idxj = match[j].source_idx;
				tar_idxj = match[j].target_idx;
				sp_j = cloud_src->points[src_idxj];
				tp_j = cloud_tar->points[tar_idxj];
				src_normalj = normals_src->points[src_idxj];
				tar_normalj = normals_tar->points[tar_idxj];
				//计算兼容性值
				compatibility_dist = getCompatibility(sp_i, sp_j, tp_i, tp_j, src_normali, src_normalj, tar_normali, tar_normalj, resolution);
				if (compatibility_dist >= threshod)
				{
					_nodes[i].degree++;
					_adjacent_matrix[i][j] = 1;
					_compatibility_matrix[i][j] = compatibility_dist;
				}
			}
		}
	}
}

void Graph::computeCircle()
{
	std::set<std::vector<int>>circleSet;
	for (int i = 0; i < _nodes.size(); i++)
	{
		std::vector<int>temp;
		for (int j = 0; j < _nodes.size(); j++)
		{
			if (_adjacent_matrix[i][j] == 1)temp.push_back(j);
		}
		for (int j : temp)
		{
			for (int k : temp)
			{
				if (_adjacent_matrix[j][k] == 1)
				{
					std::vector<int>t;
					t.push_back(i);
					t.push_back(j);
					t.push_back(k);
					sort(t.begin(), t.end());
					circleSet.insert(t);
				}
			}
		}
	}
	for (std::set<std::vector<int>>::iterator it = circleSet.begin(); it != circleSet.end(); ++it)
	{
		Circle c;
		c.index1 = (*it)[0];
		c.index2 = (*it)[1];
		c.index3 = (*it)[2];
		c.degree = _nodes[c.index1].degree + _nodes[c.index2].degree + _nodes[c.index3].degree;
		c.compatibility_dist = _compatibility_matrix[c.index1][c.index2] + _compatibility_matrix[c.index1][c.index3] + 
			_compatibility_matrix[c.index2][c.index3];
		_circles.push_back(c);
	}
	sort(_circles.begin(), _circles.end(), [](Circle c1, Circle c2) {return c1.compatibility_dist > c2.compatibility_dist; });
}

void Graph::computeLine()
{
	for (int i = 0; i < _nodes.size(); i++)
	{
		for (int j = 0; j < i; j++)
		{
			Line l;;
			l.index1 = _nodes[i].index;
			l.index2 = _nodes[j].index;
			l.degree = _nodes[l.index1].degree + _nodes[l.index2].degree;
			l.compatibility_dist = _compatibility_matrix[l.index1][l.index2];
			_lines.push_back(l);
		}
	}
	sort(_lines.begin(), _lines.end(), [](Line l1, Line l2) {return l1.compatibility_dist > l2.compatibility_dist; });
}

void Graph::computeTriple(int topK)
{
	for (int i = 0; i < topK; i++)
	{
		for (int j = 0; j < i; j++)
		{
			for (int k = 0; k < j; k++)
			{
				Triple triple;
				triple.index1 = _nodes[i].index;
				triple.index2 = _nodes[j].index;
				triple.index3 = _nodes[k].index;
				triple.degree = _nodes[i].degree + _nodes[j].degree + _nodes[k].degree;
				_triples.push_back(triple);
			}
		}
	}
	sort(_triples.begin(), _triples.end(), [](Triple t1, Triple t2) {return t1.degree > t2.degree; });
}

std::vector<Circle>Graph::getCircles()
{
	return _circles;
}
std::vector<Triple>Graph::getTriples()
{
	return _triples;
}

//求平方
double sac_cot::Square(float x)
{
	return x * x;
}
//两点之间的距离
double sac_cot::Distance(pcl::PointXYZ &A, pcl::PointXYZ &B)
{
	float result;
	result = sqrt(Square(A.x - B.x) + Square(A.y - B.y) + Square(A.z - B.z));
	return result;
}
//刚性约束项r(ci,cj)
float sac_cot::Rigidity(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j)
{
	float result;
	result = abs(Distance(source_i, source_j) - Distance(target_i, target_j));
	return result;
}
//法向量夹角
double sac_cot::NormalDistance(pcl::Normal &n_i, pcl::Normal &n_j)
{
	float A, B;
	float degree;
	A = n_i.normal_x*n_j.normal_x + n_i.normal_y*n_j.normal_y + n_i.normal_z*n_j.normal_z;
	B = (sqrt(Square(n_i.normal_x) + Square(n_i.normal_y) + Square(n_i.normal_z)))*(sqrt(Square(n_j.normal_x) + Square(n_j.normal_y) + Square(n_j.normal_z)));
	degree = 180 / M_PI * acos(A / B);
	return degree;
}
//兼容性值
double sac_cot::getCompatibility(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j,
	pcl::Normal &ns_i, pcl::Normal &ns_j, pcl::Normal &nt_i, pcl::Normal &nt_j, float resolution)
{
	float r, n, R, N;
	float compatibility;
	float a, b;
	r = Rigidity(source_i, source_j, target_i, target_j);
	R = Square(r);
	a = Square(10 * resolution);
	n = abs(NormalDistance(ns_i, ns_j) - NormalDistance(nt_i, nt_j));
	N = Square(n);
	b = Square(10);
	//compatibility = exp(-(R/a)-(N/b));
	compatibility = exp(-(R / a));
	return compatibility;
}