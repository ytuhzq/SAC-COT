#pragma once
namespace sac_cot
{
	typedef struct {
		float x;
		float y;
		float z;
	}Vertex;

	typedef struct {
		int pointID;
		Vertex x_axis;
		Vertex y_axis;
		Vertex z_axis;
	}LRF;

	typedef struct {
		int source_idx;
		int target_idx;
		LRF source_LRF;
		LRF target_LRF;
		float ratio;
		float dist;//nearest neighbor distance ratio 
	}Match_pair;

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
		int index3;
		int degree;
	}Triple;

	typedef struct {
		int index1;
		int index2;
		int degree;
		double compatibility_dist;
	}Line;

	typedef struct {
		int index;//��index��ƥ��
		int degree;//��ƥ��Ķ�
	}Node;
}
