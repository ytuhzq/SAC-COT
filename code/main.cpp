#include"CApp.h"
#include"utils.h"
#include"ransac.h"
#include<fstream>

using namespace std;

int main(int argc, char *argv[])
{
	if (argc != 4)
	{
		printf("Usage ::\n");
		printf("%s [dataload path] [data path] [out path]\n", argv[0]);
		printf("examples: .\\data\\in\\dataload.txt .\\data\\in\\ .\\data\\out\\");
		return 0;
	}
	int num = -1;
	ifstream infile(argv[1]);
	if (!infile.is_open())
	{
		cout << "未成功打开文件" << endl;
		system("pause");
	}
	infile >> num;
	while (num != 0)
	{
		num--;
		string src, tar;
		Eigen::Matrix4f gt;

		// read GT metric
		infile >> src >> tar >>
			gt(0, 0) >> gt(0, 1) >> gt(0, 2) >> gt(0, 3) >>
			gt(1, 0) >> gt(1, 1) >> gt(1, 2) >> gt(1, 3) >>
			gt(2, 0) >> gt(2, 1) >> gt(2, 2) >> gt(2, 3) >>
			gt(3, 0) >> gt(3, 1) >> gt(3, 2) >> gt(3, 3);

		sac_cot::CApp app;
		app.XYZ_Read((string(argv[2]) + src).c_str());
		app.XYZ_Read((string(argv[2]) + tar).c_str());
		// downsampling size
		app.downSampling(2.0);
		app.computeNormals();
		// key point support radius
		app.getHarrisKeypoint3D(2.0);
		// descriptor support radius
		app.SHOT_compute(15);
		// top number match
		app.calculateTopKMatchByRatio(100);
		// compatibility threshold
		app.constructGraph(0.9);
		sac_cot::BaseRansac* saccot = new sac_cot::SAC_COT();
		app.ransac(saccot);
		app.XYZ_Save((string(argv[3])).c_str(), src.substr(0, src.size() - 4).c_str(), tar.substr(0, tar.size() - 4).c_str()) ;
		//system("pause");
	}
	system("pause");
	return 0;
}