#include"header.h"
//Square 
double Square(float x)
{
	return x * x;
}
//Calculate the distance between two points
double Distance(pcl::PointXYZ &A, pcl::PointXYZ &B)
{
	float result;
	result = sqrt(Square(A.x - B.x) + Square(A.y - B.y) + Square(A.z - B.z));
	return result;
}
//r(ci,cj)
float Rigidity(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j)
{
	float result;
	result = abs(Distance(source_i, source_j) - Distance(target_i, target_j));
	return result;
}
//Normal angle
double NormalDistance(pcl::Normal &n_i, pcl::Normal &n_j)
{
	float A, B;
	float degree;
	A = n_i.normal_x*n_j.normal_x + n_i.normal_y*n_j.normal_y + n_i.normal_z*n_j.normal_z;
	B = (sqrt(Square(n_i.normal_x) + Square(n_i.normal_y) + Square(n_i.normal_z)))*(sqrt(Square(n_j.normal_x) + Square(n_j.normal_y) + Square(n_j.normal_z)));
	degree = 180 / M_PI * acos(A / B);
	return degree;
}
//Compatibility value
double getCompatibility(pcl::PointXYZ &source_i, pcl::PointXYZ &source_j, pcl::PointXYZ &target_i, pcl::PointXYZ &target_j, 
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
