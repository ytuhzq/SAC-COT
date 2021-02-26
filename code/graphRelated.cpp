#include"header.h"
using namespace std;
vector<Circle> getCircle(vector<vector<int>>matrix, int n, vector<Node>& nodes, vector<vector<double>>& compatibility_matrix)
{
	set<vector<int>>circleSet;
	vector<Circle> circles;
	for (int i = 0; i < n; i++)
	{
		vector<int>temp;
		for (int j = 0; j < n; j++)
		{
			if (matrix[i][j] == 1)temp.push_back(j);
		}
		for (int j : temp)
		{
			for (int k : temp)
			{
				if (matrix[j][k] == 1)
				{
					vector<int>t;
					t.push_back(i);
					t.push_back(j);
					t.push_back(k);
					sort(t.begin(), t.end());
					circleSet.insert(t);
				}
			}
		}
	}
	for (set<vector<int>>::iterator it = circleSet.begin(); it != circleSet.end(); ++it)
	{
		Circle c;
		c.index1 = (*it)[0];
		c.index2 = (*it)[1];
		c.index3 = (*it)[2];
		c.degree = nodes[c.index1].degree + nodes[c.index2].degree + nodes[c.index3].degree;
		c.compatibility_dist = compatibility_matrix[c.index1][c.index2] + compatibility_matrix[c.index1][c.index3] + compatibility_matrix[c.index2][c.index3];
		circles.push_back(c);
	}
	return circles;
}


vector<Circle> getCliqueCircle(vector<int>maxClique, vector<vector<double>>& compatibility_matrix)
{
	set<vector<int>>circleSet;
	vector<Circle> circles;
	int n = maxClique.size();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < i; j++)
		{
			for (int k = 0; k < j; k++)
			{
				vector<int>t;
				t.push_back(i);
				t.push_back(j);
				t.push_back(k);
				sort(t.begin(), t.end());
				circleSet.insert(t);
			}
		}
	}
	for (set<vector<int>>::iterator it = circleSet.begin(); it != circleSet.end(); ++it)
	{
		Circle c;
		c.index1 = (*it)[0];
		c.index2 = (*it)[1];
		c.index3 = (*it)[2];
		c.compatibility_dist = compatibility_matrix[c.index1][c.index2] + compatibility_matrix[c.index1][c.index3] + compatibility_matrix[c.index2][c.index3];
		circles.push_back(c);
	}
	return circles;
}

vector<Line> getLine(vector<vector<int>>&matrix, vector<Node>& nodes, vector<vector<double>>& compatibility_matrix, int n)
{
	vector<Line> line;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < i; j++)
		{
			Line l;;
			l.index1 = nodes[i].index;
			l.index2 = nodes[j].index;
			l.degree = nodes[l.index1].degree + nodes[l.index2].degree;
			l.compatibility_dist = compatibility_matrix[l.index1][l.index2];
			line.push_back(l);
		}
	}
	return line;
}

vector<Triple>getTriple(vector<Node>& nodes, int topK)
{
	vector<Triple>triples;
	for (int i = 0; i < topK; i++)
	{
		for (int j = 0; j < i; j++)
		{
			for (int k = 0; k < j; k++)
			{
				Triple triple;
				triple.index1 = nodes[i].index;
				triple.index2 = nodes[j].index;
				triple.index3 = nodes[k].index;
				triple.degree = nodes[i].degree + nodes[j].degree + nodes[k].degree;
				triples.push_back(triple);
			}
		}
	}
	return triples;
}


void computeMatiax(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, vector<Match_pair>  match, pcl::PointCloud<pcl::Normal>::Ptr normals_src,
	pcl::PointCloud<pcl::Normal>::Ptr normals_tar, double threshod, vector<Node>& nodes, vector<vector<int>>& adjacent_matrix,
	vector<vector<double>>& compatibility_matrix, float resolution)
{
	int src_idxi, tar_idxi, src_idxj, tar_idxj;
	pcl::Normal src_normali, tar_normali, src_normalj, tar_normalj;
	pcl::PointXYZ sp_i, sp_j, tp_i, tp_j;
	double compatibility_dist;

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
					nodes[i].degree++;
					adjacent_matrix[i][j] = 1;
					compatibility_matrix[i][j] = compatibility_dist;
				}
			}
		}
	}
}