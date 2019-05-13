#include "slamBase.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/common/transforms.h>

//int main()//大开眼界，可以没有main函数
//{
typedef pcl::PointXYZRGBA PointT;
int frameidd=0;

PointCloud::Ptr image2PointCloud(cv::Mat& rgb,cv::Mat& depth,CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr cloud(new PointCloud);

    for(int m=0;m<depth.rows;m++)
    {
        for(int n=0;n<depth.cols;n++)
        {
            ushort d=depth.ptr<ushort>(m)[n];
            if(d==0)
               continue;
            PointT p;

            p.z=double(d)/camera.scale;
            p.x=(n-camera.cx)*p.z/camera.fx;
            p.y=(m-camera.cy)*p.z/camera.fy;

            p.b=rgb.ptr<uchar>(m)[n*3];
            p.g=rgb.ptr<uchar>(m)[n*3+1];
            p.r=rgb.ptr<uchar>(m)[n*3+2];

            cloud->points.push_back(p);
        }

    }

        cloud->height=1;
        cloud->width=cloud->points.size();
        cloud->is_dense=false;

        return cloud;
}

cv::Point3f point2dTo3d(cv::Point3f& point,CAMERA_INTRINSIC_PARAMETERS& camera)
{
    cv::Point3f p;
    p.z=double(point.z)/camera.scale;
    p.x=(point.x-camera.cx)*p.z/camera.fx;
    p.y=(point.y-camera.cy)*p.z/camera.fy;
    return p;
}

//return 0;
//}
//同时提取特征点和描述子
void computeKeyPointsAndDesp(FRAME& frame,string detector,string descriptor)
{
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor>_descriptor;

    cv::initModule_nonfree();
    _detector=cv::FeatureDetector::create(detector.c_str());
    _descriptor=cv::DescriptorExtractor::create(descriptor.c_str());

    if(!_detector|| !_descriptor)
    {
        cerr<<"Unknown detector or descriptor type! "<<detector<<","<<descriptor<<endl;
        return;
    }
   
   _detector->detect(frame.rgb,frame.kp);
   _descriptor->compute(frame.rgb,frame.kp,frame.desp);

   return;

}


//计算两帧之间运动
RESULT_OF_PNP estimateMotion(FRAME& frame1,FRAME& frame2,CAMERA_INTRINSIC_PARAMETERS& camera)
{
    static ParameterReader pd;
    vector<cv::DMatch> matches;
    //cv::FlanBasedMatcher matcher;
    cv::FlannBasedMatcher matcher;
    matcher.match(frame1.desp,frame2.desp,matches);

    cout<<"find total "<<matches.size()<<"matches."<<endl;
    RESULT_OF_PNP result;
    vector<cv::DMatch> goodMatches;
    double minDis=9999;
    double good_match_threshold=atof(pd.getData("good_match_threshold").c_str());

    for(size_t i=0;i<matches.size();i++)
    {
        if(matches[i].distance<minDis)
            minDis=matches[i].distance;
    }
    
    //////7
    if ( minDis < 10 ) 
        minDis = 10;

    ///////////////
    for(size_t i=0;i<matches.size();i++)
    {
        if(matches[i].distance<good_match_threshold*minDis)
            goodMatches.push_back(matches[i]);
    }

    ///////7
    //这里防止匹配为0时候，无法计算pnp程序直接死掉
    if (goodMatches.size() <= 5) 
    {
        result.inliers = -1;
        return result;
    }
    //////////
     frameidd++;
    cout<<"good matches: "<<goodMatches.size()<<endl;
    cout<<frameidd<<endl;
    //第一帧的3D点
    vector<cv::Point3f> pts_obj;
    //第二帧的图像点
    vector<cv::Point2f> pts_img;

    for(size_t i=0;i<goodMatches.size();i++)
    {
        cv::Point2f p=frame1.kp[goodMatches[i].queryIdx].pt;

        ushort d=frame1.depth.ptr<ushort>(int(p.y))[int(p.x)];
        if(d==0)
           continue;
        pts_img.push_back(cv::Point2f(frame2.kp[goodMatches[i].trainIdx].pt));

        cv::Point3f pt(p.x,p.y,d);
        cv::Point3f pd=point2dTo3d(pt,camera);
        pts_obj.push_back(pd);
    }
    /////////7
    if (pts_obj.size() ==0 || pts_img.size()==0)
    {
        result.inliers = -1;
        return result;
    }
    /////////////////


    double camera_matrix_data[3][3]={
        {camera.fx,0,camera.cx},
        {0,camera.fy,camera.cy},
        {0,0,1}
    };

    cout<<"solving pnp"<<endl;
     
    cv::Mat cameraMatrix(3,3,CV_64F,camera_matrix_data);
    cv::Mat rvec, tvec, inliers;

    cv::solvePnPRansac(pts_obj,pts_img,cameraMatrix,cv::Mat(),rvec,tvec,false,100,1.0,100,inliers);

   // RESULT_OF_PNP result;
    result.rvec=rvec;
    result.tvec=tvec;
    result.inliers=inliers.rows;

    return result;

}


//cv的旋转矢量与位移矢量转换为变换矩阵
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec,cv::Mat& tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec,R);
    Eigen::Matrix3d r;
    //cv::cv2eigen(R,r);
     cv::cv2eigen(R, r);

    //平移和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    Eigen::Translation<double,3> trans(tvec.at<double>(0,0),tvec.at<double>(0,1),
    tvec.at<double>(0,2));
    T=angle;
    T(0,3)=tvec.at<double>(0,0);
    T(1,3)=tvec.at<double>(0,1);
    T(2,3)=tvec.at<double>(0,2);
    return T;
}


//将新帧合并到旧的点云里
//输入原始点云，新来的帧以及他的位姿，输出将新来的帧加到原始帧后的图像
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original,FRAME& newFrame,
Eigen::Isometry3d T,CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr newCloud=image2PointCloud(newFrame.rgb,newFrame.depth,camera);

    //合并点云
    PointCloud::Ptr output(new PointCloud());
    pcl::transformPointCloud(*original,*output,T.matrix());
    *newCloud+=*output;

    //voxel grid滤波
    //static pcl::VoxelGrid<PointT> voxel;
    pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridsize=atof(pd.getData("voxel_grid").c_str());
    voxel.setLeafSize(gridsize,gridsize,gridsize);
    voxel.setInputCloud(newCloud);
    PointCloud::Ptr tmp(new PointCloud());
    voxel.filter(*tmp);
    return tmp;
}
