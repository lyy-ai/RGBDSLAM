#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slamBase.h"

//g2o
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
//#include <g20/core/block_solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
//#include <g2o/solvers/csparse/linear_solver_csparse.h>
//#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
//error: expected type-specifier  edge->setRobustKernel( new g2o::RobustKernelHuber() );
//加上下面的头文件好啦
#include <g2o/core/robust_kernel_impl.h>
//#include <g2o/core/optimization_algorithm_levenbery.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
// error: ‘PassThrough’ is not a member of ‘pcl’


//g2o定义放在前面
//typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> SlamBlockSolver;
//typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
typedef g2o::LinearSolverEigen< SlamBlockSolver::PoseMatrixType > SlamLinearSolver; 


//给定index ,读取一帧数据
FRAME readFrame(int index,ParameterReader& pd);

//度量运动大小
double normofTransform(cv::Mat rvec,cv::Mat tvec);

//检测两个帧，结果定义
enum CHECK_RESULT{NOT_MATCHED=0,TOO_FAR_AWAY,TOO_CLOSE,KEYFRAME};

// 函数声明
CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false );
//近距离检测回环
void checkNearbyLoops(vector<FRAME>& frames,FRAME& currFrame,g2o::SparseOptimizer& opti);

//随机检测回环
void checkRandomLoops(vector<FRAME>& frames,FRAME& currFrame,g2o::SparseOptimizer& opti);



int main(int argc,char** argv)
{
    ParameterReader pd;
    int startIndex=atoi(pd.getData("start_index").c_str());
    int endIndex=atoi(pd.getData("end_index").c_str());

    //初始化
    cout<<"Initiallizing........."<<endl;
      // 所有的关键帧都放在了这里
    vector< FRAME > keyframes; 
    int currIndex = startIndex; // 当前索引为currIndex
    FRAME currFrame = readFrame( currIndex, pd ); // 上一帧数据
  

    //我们总是在比较currFrame和lastFrame
    string detector=pd.getData("detector");
    string descriptor=pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera=getDefaultCamera();
    computeKeyPointsAndDesp(currFrame,detector,descriptor);
    PointCloud::Ptr cloud=image2PointCloud(currFrame.rgb,currFrame.depth,camera);
   
    //是否显示点云
   // bool visualize=pd.getData("visualize_pointcloud")==string("yes");

///////////////
   // int min_inliers=atoi(pd.getData("min_inliers").c_str());
    //int max_norm=atof(pd.getData("max_norm").c_str());
//////////////////
    
    
    //初始化求解器
    SlamLinearSolver* linearSolver=new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);
    SlamBlockSolver* blockSolver=new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(blockSolver);

    g2o::SparseOptimizer globalOptimizer;
    //globalOptimizerAlgorithmLevenberg(solver);
     globalOptimizer.setAlgorithm( solver ); 

    //不要输出调试信息
    globalOptimizer.setVerbose(false);

    //向globalOptimizer增加一个顶点
    g2o::VertexSE3* v=new g2o::VertexSE3();
    v->setId(currIndex);
    //v-setEstimate(Eigen::Isometry3d::Identity());//估计为单位矩阵
    v->setEstimate( Eigen::Isometry3d::Identity() ); //估计为单位矩阵
    v->setFixed(true);//第一个顶点固定，不用优化
    globalOptimizer.addVertex(v);

    keyframes.push_back(currFrame);
    double keyframe_threshold=atof(pd.getData("keyframe_threshold").c_str());
    bool check_loop_closure=pd.getData("check_loop_closure")==string("yes");


    //int lastIndex=currIndex;//上一帧的id


    for(currIndex=startIndex+1;currIndex<endIndex;currIndex++)
    {
        cout<<"Reading files"<<currIndex<<endl;
        FRAME currFrame=readFrame(currIndex,pd);
        computeKeyPointsAndDesp(currFrame,detector,descriptor);
        
       //匹配该帧和keyframes里最后一帧
        CHECK_RESULT result=checkKeyframes(keyframes.back(),currFrame,globalOptimizer);
        switch(result)//根据匹配结果不同采取不同的策略
        {
        case NOT_MATCHED:
            //没匹配上，直接跳过
            cout<<"not enough inliers"<<endl;
            break;
        case TOO_FAR_AWAY:
             //太近了，也直接跳
             cout<<"too far away,may be an error"<<endl;
             break;
        case TOO_CLOSE:
             //
             cout<<"too close,not a keyframe"<<endl;
        case KEYFRAME:
            cout<<"this is a new keyframe"<<endl;
            //检测回环
            if(check_loop_closure)
            {
                checkNearbyLoops(keyframes,currFrame,globalOptimizer);
                checkRandomLoops(keyframes,currFrame,globalOptimizer);

            }
            keyframes.push_back(currFrame);
            break;
        
        default:
            break;
        }
    }
        

   // pcl::io::savePCDFile("../result5.pcd",*cloud);

   //优化所有的边
   //cout<<"optimizing pose graph,vertices: "<<globalOptimier.vertices().size()<<endl;
   cout<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
   globalOptimizer.save("../result_before.g2o");
   globalOptimizer.initializeOptimization();
   globalOptimizer.optimize(100);//迭代步数
   globalOptimizer.save("../result_after.g2o");
   cout<<"Optimization done "<<endl;

   //拼接点云地图
   cout<<"saving the point cloud map..."<<endl;
   PointCloud::Ptr output(new PointCloud());//全局地图
   PointCloud::Ptr tmp(new PointCloud());

   pcl::VoxelGrid<PointT> voxel;//网格滤波器
   //pcl::PassThrough<PointT> pass;//z方向区间滤波器，由于rgbd相机的有效深度有限，把太远的去掉
    pcl::PassThrough<PointT> pass;


   //pass.setFilterFileName("z");
   pass.setFilterFieldName("z");
   pass.setFilterLimits(0.0,4.0);//4cm以上就不要了

   double gridsize=atof(pd.getData("voxel_grid").c_str());//分辨率在此调整
   //voxel.setLeafSie(gridsize,gridsize,gridsize);
   voxel.setLeafSize( gridsize, gridsize, gridsize );

   for(size_t i=0;i<keyframes.size();i++)
   {
       //从g2o里取出一帧
       //g2o::VertexSE3* vertex=dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(
         //  keyframes[i].frameID
     //  ));
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].frameID ));
       Eigen::Isometry3d pose=vertex->estimate();//该帧优化后的位姿
       PointCloud::Ptr newCloud=image2PointCloud(keyframes[i].rgb,keyframes[i].depth,camera);//转成点云
       //开始滤波
       voxel.setInputCloud(newCloud);
       voxel.filter(*tmp);
       pass.setInputCloud(tmp);
       pass.filter(*newCloud);

       //把点云变换后加入到全局地图
       pcl::transformPointCloud(*newCloud,*tmp,pose.matrix());
       *output+=*tmp;
       tmp->clear();
       newCloud->clear();
   }

   voxel.setInputCloud(output);
   voxel.filter(*tmp);

   pcl::io::savePCDFile("../result7.pcd",*tmp);

   cout<<"final map is saved "<<endl;
   globalOptimizer.clear();
    return 0;
}


FRAME readFrame(int index,ParameterReader& pd)
{
    FRAME f;
    string rgbDir=pd.getData("rgb_dir");
    string depthDir=pd.getData("depth_dir");

    string rgbExt=pd.getData("rgb_extension");
    string depthExt=pd.getData("depth_extension");

    stringstream ss;
    ss<<rgbDir<<index<<rgbExt;
    string filename;
    ss>>filename;
    f.rgb=cv::imread(filename);

    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;

    f.depth=cv::imread(filename,-1);
    return f;
}

double normofTransform(cv::Mat rvec,cv::Mat tvec)
{
    return fabs(min(cv::norm(rvec),2*M_PI-cv::norm(rvec)))+fabs(cv::norm(tvec));
}


CHECK_RESULT checkKeyframes(FRAME& f1,FRAME& f2,g2o::SparseOptimizer& opti,bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers=atoi(pd.getData("min_inliers").c_str());
    static double max_norm=atof(pd.getData("max_norm").c_str());
    static double keyframe_threshold=atof(pd.getData("keyframe_threshold").c_str());

   //static double max_norm_lp=atof(pd.getData("max_norm_lp").c_str);
    static double max_norm_lp=atof(pd.getData("max_norm_lp").c_str());
    static CAMERA_INTRINSIC_PARAMETERS camera=getDefaultCamera();
    
    //static g2o::RobustKernel* robustKernel=g2o::RobustKernelFactory::instance()->construct("Cauchy");

    //比较f1和f2
    RESULT_OF_PNP result=estimateMotion(f1,f2,camera);
    if(result.inliers<min_inliers)//inliers不够，放弃该帧
        return NOT_MATCHED;
    
    //计算运动范围是否太大
   // double norm=normfTransform(result.rvec,result.tvec);
   double norm = normofTransform(result.rvec, result.tvec);
    if(is_loops==false)
    {
        if(norm>=max_norm)
            return TOO_FAR_AWAY;//太远错误
    }
    else
    {
        if(norm>=max_norm_lp)
            return TOO_FAR_AWAY;
    }
   
   //向g2o中增加这个顶点与上一帧联系的边
   //顶点部分只要设定id即好
   if(is_loops==false)
   {
       g2o::VertexSE3 *v=new g2o::VertexSE3();
      // v->setID(f2.frameID);
       v->setId( f2.frameID );
       v->setEstimate(Eigen::Isometry3d::Identity());
       opti.addVertex(v);
   }

        //边部分
        g2o::EdgeSE3* edge=new g2o::EdgeSE3();
        //连接此边的两个顶点id
       // edge->vertices()[0]=globalOptimizer.vertex(f1.frameID);
        edge->setVertex( 0, opti.vertex(f1.frameID ));
        //edge->vertices()[1]=globalOptimizer.vertex(f2.frameID);
          edge->setVertex( 1, opti.vertex(f2.frameID ));
         
         ///////////////////////////
          edge->setRobustKernel( new g2o::RobustKernelHuber() );
       // edge->setRobustKernel(robustKernel);
///////////////////////////////////////////
        //信息矩阵
        Eigen::Matrix<double,6,6> information=Eigen::Matrix<double,6,6>::Identity();

         // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
        // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
         // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
         information(0,0)=information(1,1)=information(2,2)=100;
         information(3,3)=information(4,4)=information(5,5)=100;

         //也可以将角度设大一些，表示对角度的估计更加准确
         edge->setInformation(information);

         //边的估计是pnp求解的结果
         Eigen::Isometry3d T=cvMat2Eigen(result.rvec,result.tvec);
         //edge->setMeasurement(T.inverse);
          edge->setMeasurement( T.inverse() );
         //将边加入图中
         opti.addEdge(edge);

         return KEYFRAME;
}


//void checkNearbyLoops(vector<FRAME&> frames,FRAME& currFrame,g2o::SparseOptimizer&
//opti)
void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti )
{
    static ParameterReader pd;
    static int nearby_loops=atoi(pd.getData("nearby_loops").c_str());

    //把当前帧currFrame和frames里末尾几个侧一遍
   // if(frames.size()<=nearby_loops())
    if(frames.size()<=nearby_loops)
    {
        //没有足够的关键性帧，检查每一个
        for(size_t i=0;i<frames.size();i++)
        {
            checkKeyframes(frames[i],currFrame,opti,true);
        }
    }

    else
    {
        //检查最近
        for(size_t i=frames.size()-nearby_loops;i<frames.size();i++)
        {
            checkKeyframes(frames[i],currFrame,opti,true);
        }
    }
}


//void checkRandomLoops(vector<FRAME&> frames,FRAME& currFrame,g2o::SparseOptimizer&
//opti)
void checkRandomLoops(vector<FRAME>& frames,FRAME& currFrame,g2o::SparseOptimizer&
opti)
{
    static ParameterReader pd;
    static int random_loops=atoi(pd.getData("random_loops").c_str());

    //随机取一些帧进行检测
    if(frames.size()<=random_loops)
    {
        //没有足够的关键性帧，检查每一个
        for(size_t i=0;i<frames.size();i++)
        {
            checkKeyframes(frames[i],currFrame,opti,true);
        }
    }

    else
    {
        //检查最近
        for(size_t i=0;i<random_loops;i++)
        {   
            int index=rand()%frames.size();
            checkKeyframes(frames[index],currFrame,opti,true);
        }
    }
}
