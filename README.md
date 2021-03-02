# SAC-COT
C++ implementation of the SAC-COT method proposed by J. Yang et al. [1]

# Running environment：
VS2017(https://visualstudio.microsoft.com/zh-hans/.)  
PCL1.81(https://github.com/PointCloudLibrary/pcl/releases.)

# Run：
It can be run directly in the Main function, or it can be compiled and run on the command line  

# Directories:：
data：Contains a set of test data, including the source point cloud, the target point cloud, and the ground truth matrix
code：Program source code  
    Main:The main function  
    header.h:Header file  
    PclBasic.cpp：Point cloud processing, such as computing key points, descriptors, point cloud resolution, correspondence...  
    Compatibility.cpp:Calculates compatibility values between correspondences  
    graphRelated.cpp：According to the compatibility value between correspondences, calculate the compatibility matrix and the compatibility triangle.   
    RANSACRelated.cpp：Functions related with RANSAC  

# Main function flow
1、 Read the point cloud data and the GT matrix  
2、 Compute the point cloud resolution and downsample the Point Cloud (in order to reduce the size of the point cloud and increase the speed)  
3、 Compute Normals, key points, descriptors (the key points used in the program are Harris 3D, the descriptors are SHOT, you can use other key points + descriptors combination)  
4、 Calculate the initial matching set (based on the distance of the descriptor)  
5、 Calculated compatibility values between the matches, and construct the ternary ring and the compatibility triangle  
6、 Sorting ternary rings and compatibility triangles, guiding RANSAC sampling  

# Pipeline of the proposed SAC-COT estimator.

# Bibliography:  
[1] J. Yang, Z. Huang, S. Quan, Z. Qi and Y. Zhang, "SAC-COT: Sample Consensus by Sampling Compatibility Triangles in Graphs for 3-D Point Cloud Registration," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3058552.
