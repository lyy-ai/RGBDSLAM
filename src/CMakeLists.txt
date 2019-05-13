#增加一个可执行文件
ADD_EXECUTABLE(main main.cpp)

SET(OpenCV_DIR "/home/lyy/Linux_dependence/opencv-2.4.9/build")
#FIND_PACKAGE(PCL REQUIRED)
FIND_PACKAGE(OpenCV  REQUIRED)

# 增加PCL库的依赖
#FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization )
#对‘pcl::VoxelGrid<pcl::PointXYZRGBA>::applyFilter(pcl::PointCloud<pcl::PointXYZRGBA>&)’未定义的引用
#因为没有加入filters，所以报上面的错误

FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04
#include_directories( "/usr/include/pcl-1.9/" )
#add_definitions( "/usr/share/pcl-1.9/" )
#link_directories("/usr/lib/libpcl_common.so" "/usr/lib/libpcl_io.so")
#list (REMOVE_ITEM PCL_LIBRARIES "vtkproj4")


# 添加g2o的依赖
# 因为g2o不是常用库，要添加它的findg2o.cmake文件
LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O )
# CSparse
FIND_PACKAGE( CSparse )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )


#添加头文件和库文件
ADD_DEFINITIONS(${PCL_DEFINTIONS})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
LINK_LIBRARIES(${PCL_LIBRARY_DIRS})

INCLUDE_DIRECTORIES("usr/include/eigen3")

ADD_EXECUTABLE(generatePointCloud generatePointCloud.cpp)
TARGET_LINK_LIBRARIES(generatePointCloud ${OpenCV_LIBS} ${PCL_LIBRARIES})

#ADD_EXECUTABLE(slambase slamBase.cpp)
#如果没有ADD_LIBRARY导至无法生成target(slambase),因此报错后面main未定义
#TARGET_LINK_LIBRARIES(slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})


ADD_LIBRARY( slambase slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )



#ADD_EXECUTABLE(detectFeatures detectFeatures.cpp)
#[100%] Linking CXX executable ../../bin/detectFeatures
#CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o：在函数‘main’中：
#detectFeatures.cpp:(.text+0xf61)：对‘point2dTo3d(cv::Point3_<float>&, CAMERA_INTRINSIC_PARAMETERS&)’未定义的引用


#ADD_LIBRARY(detectFeatures detectFeatures.cpp)
ADD_EXECUTABLE(detectFeatures detectFeatures.cpp)
#TARGET_LINK_LIBRARIES(detectFeatures ${OpenCV_LIBS} ${PCL_LIBRARIES} )
#TARGET_LINK_LIBRARIES(detectFeatures ${OpenCV_LIBS} ${PCL_LIBRARIES} libslambase.a )#也行

TARGET_LINK_LIBRARIES(detectFeatures slambase ${OpenCV_LIBS} ${PCL_LIBRARIES}  )

ADD_EXECUTABLE(jointPointCloud jointPointCloud.cpp)
TARGET_LINK_LIBRARIES(jointPointCloud slambase ${OpenCV_LIBS} ${PCL_LIBRARIES} )

#5
ADD_EXECUTABLE( visualOdometry visualOdometry.cpp)
TARGET_LINK_LIBRARIES( visualOdometry
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

#6
ADD_EXECUTABLE( slamEnd slamEnd.cpp)
#TARGET_LINK_LIBRARIES( slamEnd
  #  slambase
  # ${OpenCV_LIBS} 
   # ${PCL_LIBRARIES} 
    # g2o_core g2o_stuff
    # )
#仅仅上面报错如下
#slamEnd.cpp:(.text+0xac6)：对‘g2o::VertexSE3::VertexSE3()’未定义的引用
#slamEnd.cpp:(.text+0xdb9)：对‘g2o::VertexSE3::VertexSE3()’未定义的引用
#slamEnd.cpp:(.text+0xe3b)：对‘g2o::EdgeSE3::EdgeSE3()’未定义的引用
#故修改如下
TARGET_LINK_LIBRARIES( slamEnd
    slambase
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})


#7
ADD_EXECUTABLE( slam slam.cpp)
TARGET_LINK_LIBRARIES( slam
    slambase
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})
