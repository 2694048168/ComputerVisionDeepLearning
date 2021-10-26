/**
 * @File    : roberts_algorithm.cpp
 * @Brief   : 基于方向差分卷积核进行卷积操作——Roberts Operator(Algorithm)
 * @Author  : Wei Li
 * @Date    : 2021-10-13
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void RobertsOperator(cv::InputArray src, cv::OutputArray dst, int ddepth, int x=1, int y=0, int borderType=cv::BORDER_DEFAULT)
{
    CV_Assert(!(x==0 && y==0));
    cv::Mat roberts_kernel_1 = (cv::Mat_<float>(2, 2) << 1, 0, 0, -1);
    cv::Mat roberts_kernel_2 = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);

    // 当 x 不等于 0 时候，src 和 roberts_kernel_1 卷积
    if (x != 0 && y == 0)
    {
        cv::Mat kernel_1_filp;
        cv::flip(roberts_kernel_1, kernel_1_filp, -1);
        cv::filter2D(src, dst, ddepth, kernel_1_filp, cv::Point(0, 0), borderType);
    }
    
    // 当 y 不等于 0 时候，src 和 roberts_kernel_2 卷积
    if (y != 0 && x == 0)
    {
        cv::Mat kernel_2_filp;
        cv::flip(roberts_kernel_2, kernel_2_filp, -1);
        cv::filter2D(src, dst, ddepth, kernel_2_filp, cv::Point(0, 0), borderType);
    }
}

// -------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread(argv[1], 0);
    if (!image.data)
    {
        std::cout << "Error: Reading image file unsuccessfully.";
        return -1;
    }
    cv::imshow("OriginImage", image);

    // 原始图像和 Roberts kernerl-1 卷积
    cv::Mat img_roberts_1;
    RobertsOperator(image, img_roberts_1, CV_32FC1, 1, 0);

    // 原始图像和 Roberts kernerl-2 卷积
    cv::Mat img_roberts_2;
    RobertsOperator(image, img_roberts_2, CV_32FC1, 0, 1);

    // 两个卷积结果的灰度级显示
    cv::Mat abs_img_roberts_1, abs_img_roberts_2;
    cv::convertScaleAbs(img_roberts_1, abs_img_roberts_1, 1, 0);
    cv::convertScaleAbs(img_roberts_2, abs_img_roberts_2, 1, 0);
    cv::imshow("135_edge", abs_img_roberts_1);
    cv::imshow("45_edge", abs_img_roberts_2);

    // 根据 roberts 两个卷积结果，获取最终图像的边缘强度
    cv::Mat edge;
    cv::Mat img_roberts_1_2, img_roberts_2_2;
    cv::pow(img_roberts_1, 2.0, img_roberts_1_2);
    cv::pow(img_roberts_2, 2.0, img_roberts_2_2);
    cv::sqrt(img_roberts_1_2 + img_roberts_2_2, edge);
    // 数据类型转换，边缘强度的灰度级显示
    edge.convertTo(edge, CV_8UC1);
    cv::imshow("edge", edge);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
