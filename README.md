# SAC-COT
论文SAC-COT: Sample Consensus by Sampling Compatibility Triangles in Graphs for 3-D Point Cloud Registration的方法代码

环境配置：VS2017 + PCL1.81

文件说明：
data：包含一组测试用的数据，含源点云、目标点云、ground truth矩阵
code：算法代码
    Main:程序入口
    header.h:头文件
    PclBasic.cpp：主要对点云进行处理，如计算关键点，描述子、点云分辨率、匹配等
    Compatibility.cpp计算匹配之间的兼容性值
    graphRelated.cpp：根据匹配之间的兼容性值，计算兼容性矩阵和兼容三角形
    Visualization.cpp：可视化相关代码（本程序并未使用）
    
运行：可直接在Main函数中运行，也可以编译后在命令行中运行

算法流程（和Main函数一致）
1、 读取点云数据和GT矩阵
2、 计算点云分辨率，对点云进行降采样（为了减小点云规模，提高速度）
3、 计算法线、关键点、描述子（程序中使用的关键点为Harris3D，描述子为SHOT，可以使用其它关键点+描述子组合）
4、 计算初始匹配集（根据描述子对应距离计算）
5、 计算匹配之间的兼容性值，构建三元环和兼容性三角形
6、 将三元环和兼容性三角形排序，指导RANSAC采样
