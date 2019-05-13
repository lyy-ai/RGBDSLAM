#pragma once

#include <fstream>
#include <vector>
using namespace std;

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/opencv.hpp>
//error: ‘KeyPoint’ is not a member of ‘cv’
//加上#include <opencv2/opencv.hpp>就好啦

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/io.h>
#include <pcl/filters/statistical_outlier_removal.h>




//类型定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//3
//相机内参结构
struct CAMERA_INTRINSIC_PARAMETERS 
{ 
    double cx,cy,fx,fy,scale;
};//这个分号忘记了下一行报错

//4
//帧结构
struct FRAME
{   
    int frameID; //7里加的
    cv::Mat rgb,depth;
    cv::Mat desp;//描述子
    vector<cv::KeyPoint> kp;//关键点
};

//pnp结果
struct RESULT_OF_PNP
{
    cv::Mat rvec,tvec;
    int inliers;
};

//读取参数类
class ParameterReader
{
    public:
        ParameterReader(string filename="/home/lyy/00SLAM/RGBDSLAM/parameters.txt")
        {
            ifstream fin(filename.c_str());
            if(!fin)
            {
                cerr<<"parameter file does not exist "<<endl;
                return;
            }
            while(!fin.eof())
            {
                string str;
                getline(fin,str);
                if(str[0]=='#')
                    continue;
                
                int pos=str.find("=");
                if(pos==-1)
                    continue;
                string key=str.substr(0,pos);
                string value=str.substr(pos+1,str.length());
                data[key]=value;

                if(!fin.good())
                    break;
            }
        }

        string getData(string key)
        {
            map<string,string>::iterator iter=data.find(key);
            if(iter==data.end())
            {
                cerr<<"Parameter name"<<key<<"not found!"<<endl;
                return string("NOT_FOUND");
            }

            return iter->second;
        }
public:
    map<string,string> data;
};


//3
//函数接口
PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS & camera );
//pcl::PointCloud<PointT>::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera );

//单个图像从图像坐标换到空间坐标
cv::Point3f point2dTo3d(cv::Point3f& point,CAMERA_INTRINSIC_PARAMETERS & camera);

//4
//函数接口
//同时提取特正点和描述子
void computeKeyPointsAndDesp(FRAME& frame,string detector,string descriptor);

//计算两帧之间的运动
RESULT_OF_PNP estimateMotion(FRAME& frame1,FRAME& frame2,CAMERA_INTRINSIC_PARAMETERS& camera);

//5

Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec,cv::Mat& tvec);
//将新帧合并到旧的点云里
//输入原始点云，新来的帧以及他的位姿，输出将新来的帧加到原始帧后的图像
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original,FRAME& newFrame,
Eigen::Isometry3d T,CAMERA_INTRINSIC_PARAMETERS& camera);

inline static CAMERA_INTRINSIC_PARAMETERS getDefaultCamera()
{
    ParameterReader pd;
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );
    return camera;
}