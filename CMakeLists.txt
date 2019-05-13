CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(RGBDslam)
#SET(CMAKE_CXX_COMPILER "g++")
SET(CMAKE_CXX_FLAGS "-std=c++11")

#设定可执行二进制文件的目录
SET(EXECUTABLE_OUTPUT_PATH "/home/lyy/00SLAM/RGBDSLAM/bin")

#设定存放编译出来文件的目录
SET(LIBRARY_OUTPUT_PATH "/home/lyy/00SLAM/RGBDSLAM/lib")

#把上术目录设为连接目录
LINK_directories("/home/lyy/00SLAM/RGBDSLAM/lib")

#设定头文件目录
INCLUDE_DIRECTORIES("/home/lyy/00SLAM/RGBDSLAM/include")

#添加子文件夹,也就是进入源代码文件夹继续构建
ADD_SUBDIRECTORY("/home/lyy/00SLAM/RGBDSLAM/src")