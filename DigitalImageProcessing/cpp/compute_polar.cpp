/**
 * @File    : compute_polar.cpp
 * @Brief   : 笛卡尔坐标和极坐标之间转换
 * @Author  : Wei Li
 * @Date    : 2021-09-15
*/

#include <iostream>
#include <cmath>

#include <opencv2/core.hpp>

const double PI = acos(-1); // arccos(-1) = π

int main(int argc, char **argv)
{
    // 1. 利用数学关系计算笛卡尔坐标和极坐标进行转换
    // 笛卡尔坐标为 (11, 13) 以 (3， 5) 为中心
    double r = std::sqrt(std::pow(11 - 3, 2) + std::pow(13 - 5, 2));
    // 圆周率 std::numbers::pi; std==C++20
    double theta = std::atan2(13 - 5, 11 - 3) / PI * 180;
    std::cout << "The r is " << r << std::endl;
    std::cout << "The theta is " << theta << std::endl;

    // 2. 利用 OpenCV 函数计算笛卡尔坐标和极坐标进行转换
    cv::Mat x = (cv::Mat_<float>(3, 3) << 0, 1, 2, 0, 1, 2, 0, 1, 2) - 1;
    cv::Mat y = (cv::Mat_<float>(3, 3) << 0, 0, 0, 1, 1, 1, 2, 2, 2) - 1;
    cv::Mat r_opencv, theta_opencv;
    cv::cartToPolar(x, y, r_opencv, theta_opencv, true);
    std::cout << "The r is with OpenCV " << r_opencv << std::endl;
    std::cout << "The theta is with OpenCV " << theta_opencv << std::endl;

    // 3. 将极坐标转换为笛卡尔坐标
    cv::Mat angle = (cv::Mat_<float>(2, 2) << 30, 31, 30, 31);
    cv::Mat r_polar = (cv::Mat_<float>(2, 2) << 10, 10, 11, 11);
    // # 计算出来的是以 原点 (0，0) 为变换中心的坐标，按照需要进行转换
    cv::Mat x_ploar, y_polar;
    cv::polarToCart(r_polar, angle, x_ploar, y_polar, true);
    x_ploar += -12;
    y_polar += 15;
    std::cout << "The x is with OpenCV " << x_ploar << std::endl;
    std::cout << "The y is with OpenCV " << y_polar << std::endl;

    return 0;
}
