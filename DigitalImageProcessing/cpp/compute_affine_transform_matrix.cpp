/**
 * @File    : compute_affine_transform_matrix.cpp
 * @Brief   : 已知原始坐标和变换后的坐标，计算仿射变换矩阵
 * @Author  : Wei Li
 * @Date    : 2021-09-12
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// C++ API of OpenCV 输出参数有两种方式
// 1. 原始坐标和对应的变换后的坐标以 Point2f 进行存储
// 2. 原始坐标和对应的变换后的坐标以 Mat 进行存储
// -------------------------------------------
// 3. 利用 Linux OS 对 OpenCV 源代码进行编译和链接，同时利用 CMake 进行构建
// 编译和链接参考：https://gitee.com/weili_yzzcq/C-and-C-plus-plus/tree/master/OpenCV_Linux_Ubuntu
// 编译和链接参考：https://github.com/2694048168/C-and-C-plus-plus/tree/master/OpenCV_Linux_Ubuntu
// -------------------------------------------
int main(int argc, char** argv)
{
    // 1. 第一种方式
    cv::Point2f src[] = {cv::Point2f(0, 0), cv::Point2f(200, 0), cv::Point2f(0, 200)};
    cv::Point2f dst[] = {cv::Point2f(0, 0), cv::Point2f(100, 0), cv::Point2f(0, 100)};
    cv::Mat affine_transform_matrix = cv::getAffineTransform(src, dst);
    std::cout << affine_transform_matrix << std::endl;

    // 2. 第二种方式
    cv::Mat src_mat = (cv::Mat_<float>(3, 2) << 0, 0, 200, 0, 0, 200);
    cv::Mat dst_mat = (cv::Mat_<float>(3, 2) << 0, 0, 100, 0, 0, 100);
    cv::Mat affine_transform_mat = cv::getAffineTransform(src_mat, dst_mat);
    std::cout << affine_transform_mat << std::endl;

    // 通过基本的仿射变换进行矩阵乘法求解仿射变换矩阵
    // 缩放矩阵
    cv::Mat scale = (cv::Mat_<float>(3, 3) << 0.5, 0, 0, 0, 0.5, 0, 0, 0, 1);
    // 平移矩阵
    cv::Mat translation = (cv::Mat_<float>(3, 3) << 1, 0, 100, 0, 1, 200, 0, 0, 1);
    cv::Mat affine_transform;
    // 注意计算方向(先缩放后平移), 仿射矩阵计算和需要操作的过程相反！！！
    // cv::gemm(scale, translation, 1.0, cv::Mat(), 0, affine_transform, 0);
    cv::gemm(translation, scale, 1.0, cv::Mat(), 0, affine_transform, 0);
    std::cout << affine_transform << std::endl;

    // 等比例进行缩放 平移 旋转操作
    // center 变换中心的坐标；angle : 等比例缩放的系数  scale : 逆时针旋转的角度，单位为角度，而不是弧度；该值为负数，即为顺时针旋转。
    cv::Mat rotataion_matrix = cv::getRotationMatrix2D(cv::Point2f(40, 50), 30, 0.5);
    std::cout << rotataion_matrix << std::endl;
    
    return 0;
}
