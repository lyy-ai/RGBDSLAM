#include <iostream>
#include <string>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

//定义点云类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//相机内参
const double camera_factor=1000;
const double camera_cx=325.5;
const double camera_cy=253.5;
const double camera_fx=518.0;
const double camera_fy=519.0;

int main(int argc,char** argv)
{
    cv::Mat rgb,depth; 
 rgb=cv::imread("../data/rgb.png");
//flags=-1表示读取原始数据不做任何修改
depth=cv::imread("../data/depth.png",-1);

//点云变量，使用智能指针，创建一个空点云，用完会自动释放
PointCloud::Ptr cloud(new PointCloud);
//遍历深度图
for(int m=0;m<depth.rows;m++)
{
    for(int n=0;n<depth.cols;n++)
    {
        //获取深度图中(m,n)处的值
        ushort d=depth.ptr<ushort>(m)[n];
        if(d==0)
           continue;
        //若d存在，则向点云增加一个点
        PointT p;

        //计算这个点的空间坐标
         p.z=double(d)/camera_factor;
         p.x=(n-camera_cx)*p.z/camera_fx;
         p.y=(m-camera_cy)*p.z/camera_fy;

         //rgb是三通道BGR格式，所有按下面的顺序获取颜色
         p.b=rgb.ptr<uchar>(m)[n*3];
         p.g=rgb.ptr<uchar>(m)[n*3+1];
         p.r=rgb.ptr<uchar>(m)[n*3+2];

         //把p加入到点云
         cloud->points.push_back(p);
    }
}
    //设置并保存点云
    cloud->height=1;
    cloud->width=cloud->points.size();
    cout<<"point cloud size= "<<cloud->points.size()<<endl;
    cloud->is_dense=false;
    pcl::io::savePCDFile("../pointcloud.pcd",*cloud);
    //清除数据并退出
    cloud->points.clear();
    return 0;

}
