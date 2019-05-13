#include <iostream>
#include <vector>
#include "slamBase.h"
using namespace std;

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
using namespace cv;

int main(int argc,char** argv)
{
   cv::Mat rgb1,rgb2,depth1,depth2;
   rgb1=cv::imread("../data/rgb1.png");
   rgb2=cv::imread("../data/rgb2.png");
   depth1=cv::imread("../data/depth1.png",-1);
   depth2=cv::imread("../data/depth2.png",-1);

   //声明特征提取与描述子提取器
   cv::Ptr<cv::FeatureDetector>_detector;
   cv::Ptr<cv::DescriptorExtractor>_descriptor;

   //构建提取器，默认两者都sift
   cv::initModule_nonfree();
   _detector=cv::FeatureDetector::create("GridSIFT");//2.4.9
   _descriptor=cv::DescriptorExtractor::create("SIFT");
   
   vector<cv::KeyPoint> kp1,kp2;//关键点
   _detector->detect(rgb1,kp1);//提取关键点
   _detector->detect(rgb2,kp2);
   
    //初始化
    //vector<KeyPoint> kp1,kp2;
   // Ptr<ORB> orb=ORB::create(500,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
   //Ptr<ORB> orb;//3.4.1
   //检测fast角点
   //orb->detect(rgb1,kp1);//提取关键点
   //orb->detect(rgb2,kp2);

   cout<<"Key points of two images: "<<kp1.size()<<","<<kp2.size()<<endl;

   //可视化，显示关键点
   cv::Mat imgShow;
   cv::drawKeypoints(rgb1,kp1,imgShow,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
   cv::imshow("keypoints",imgShow);
   cv::imwrite("../keypoints.png",imgShow);
   cv::waitKey(0);

   //计算描述子
   cv::Mat desp1,desp2;
   _descriptor->compute(rgb1,kp1,desp1);
   _descriptor->compute(rgb2,kp2,desp2);

    //orb->compute(rgb1,kp1,desp1);
    //orb->compute(rgb2,kp2,desp2);
    
    
 
   //匹配描述子
   vector<cv::DMatch> matches;
   cv::FlannBasedMatcher matcher;
    //cv::BFMatcher matcher(NORM_HAMMING);//orb
   //cv::BruteForceMatcher<L2<float>> matcher;
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");

   matcher.match(desp1,desp2,matches);
   cout<<"find total "<<matches.size()<<"matches"<<endl;

   //可视化，显示匹配特征
   cv::Mat imgMatches;
   cv::drawMatches(rgb1,kp1,rgb2,kp2,matches,imgMatches);
   cv::imshow("matches",imgMatches);
   cv::imwrite("../matches.png",imgMatches);
   cv::waitKey(0);

   //筛选匹配，把距离太大的去掉，使用准则是去掉大于4倍最小距离的匹配
   vector<cv::DMatch> goodMatches;
   double minDis=9999;
   for(size_t i=0;i<matches.size();i++)
   {
       if(matches[i].distance<minDis)
           minDis=matches[i].distance;

   }

   for(size_t i=0;i<matches.size();i++)
   {
       if(matches[i].distance<4*minDis)
           goodMatches.push_back(matches[i]);
   }

   //显示good matches
   cout<<"good matches= "<<goodMatches.size()<<endl;
   cv::drawMatches(rgb1,kp1,rgb2,kp2,goodMatches,imgMatches);
   cv::imshow("good matches",imgMatches);
   cv::imwrite("../good_matches.png",imgMatches);
   cv::waitKey(0);

   //计算图像间的运动
   //第一帧的三维点
   vector<cv::Point3f> pts_obj;
   //第二帧的图像点
   vector<cv::Point2f> pts_img;

   //相机内参
   CAMERA_INTRINSIC_PARAMETERS c;
   c.cx=325.5;
   c.cy=253.5;
   c.fx=518.0;
   c.fy=519.0;
   c.scale=1000.0;

   for(size_t i=0;i<goodMatches.size();i++)
   {
       //query是第一个，train是第二个
       cv::Point2f p=kp1[goodMatches[i].queryIdx].pt;
       ushort d=depth1.ptr<ushort>(int(p.y))[int(p.x)];
       if(d==0)
          continue;
        pts_img.push_back(cv::Point2f(kp2[goodMatches[i].trainIdx].pt));

        //将（u,v,d)转成（x,y,z)
        cv::Point3f pt(p.x,p.y,d);
        cv::Point3f pd=point2dTo3d(pt,c);
        pts_obj.push_back(pd);
   }

   double camera_matrix_data[3][3]={
       {c.fx,0,c.cx},
       {0,c.fy,c.cy},
       {0,0,1}
   };

   //构建相机矩阵
   cv::Mat cameraMatrix(3,3,CV_64F,camera_matrix_data);
   cv::Mat rvec,tvec,inliers;

   //求解pnp
   cv::solvePnPRansac(pts_obj,pts_img,cameraMatrix,cv::Mat(),rvec,tvec,false,100,1.0,100,inliers);
   cout<<"inliers: "<<inliers.rows<<endl;
   cout<<"R= "<<rvec<<endl;
   cout<<"t= "<<tvec<<endl;

   //画出inliers匹配
   vector<cv::DMatch> matchesShow;
   for(size_t i=0;i<inliers.rows;i++)
   {
       matchesShow.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
   }
   cv::drawMatches(rgb1,kp1,rgb2,kp2,matchesShow,imgMatches);
   cv::imshow("inlier matches",imgMatches);
   cv::imwrite("../inlier.png",imgMatches);
   cv::waitKey(0);

   return 0;

}
