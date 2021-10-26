/**
 * @File    : scharr_operator.cpp
 * @Brief   : Scharr 边缘检测算子 —— 不可分离卷积核
 * @Author  : Wei Li
 * @Date    : 2021-10-16
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// 卷积运算 OpenCV with CPlusPlus
void conv2D(cv::InputArray src, cv::InputArray kernel, cv::OutputArray dst, int ddepth, cv::Point anchor=cv::Point(-1, -1), int borderType=cv::BORDER_DEFAULT)
{
    // 1. 卷积核逆时针翻转 180 度
    cv::Mat kernelFlip;
    cv::flip(kernel, kernelFlip, -1);
    // 利用计算自相关函数完成卷积运算
    cv::filter2D(src, dst, ddepth, kernelFlip, anchor, 0.0, borderType);
}

void ScharrOperator(cv::InputArray src, cv::OutputArray dst, int ddepth, int x, int y=0, int borderType=cv::BORDER_DEFAULT)
{
    CV_Assert(!(x == 0 && y == 0));
    cv::Mat scharr_x = (cv::Mat_<float>(3, 3) << 3, 0, 3, 10, 0, -10, 3, 0, -3);
    cv::Mat scharr_y = (cv::Mat_<float>(3, 3) << 3, 10, 3, 0, 0, 0, -3, -10, -3);

    // 当 x != 0, src and scharr_x convolution, 卷积结果反映垂直方向上的边缘强度
    if (x != 0 && y == 0)
    {
        conv2D(src, scharr_x, dst, ddepth, cv::Point(-1, -1), cv::BORDER_DEFAULT);
    }
    
    // 当 y != 0, src and scharr_y convolution, 卷积结果反映水平方向上的边缘强度
    conv2D(src, scharr_y, dst, ddepth, cv::Point(-1, -1), cv::BORDER_DEFAULT);
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

    // 原始图像和 scharr kernerl-x 卷积
    cv::Mat img_scharr_x;
    ScharrOperator(image, img_scharr_x, CV_32FC1, 1, 0, cv::BORDER_DEFAULT);

    // 原始图像和 scharr kernerl-y 卷积
    cv::Mat img_scharr_y;
    ScharrOperator(image, img_scharr_y, CV_32FC1, 0, 1, cv::BORDER_DEFAULT);

    // 两个卷积结果的灰度级显示
    cv::Mat abs_img_scharr_x, abs_img_scharr_y;
    cv::convertScaleAbs(img_scharr_x, abs_img_scharr_x, 1, 0);
    cv::convertScaleAbs(img_scharr_y, abs_img_scharr_y, 1, 0);
    cv::imshow("horizontal_edge", abs_img_scharr_x);
    cv::imshow("vertical_edge", abs_img_scharr_y);

    // 根据 prewiit 两个卷积结果，获取最终图像的边缘强度
    cv::Mat edge;
    cv::Mat img_scharr_x_x, img_scharr_y_y;
    cv::pow(img_scharr_x, 2.0, img_scharr_x_x);
    cv::pow(img_scharr_y, 2.0, img_scharr_y_y);
    cv::sqrt(img_scharr_x_x + img_scharr_y_y, edge);
    // 数据类型转换，边缘强度的灰度级显示
    edge.convertTo(edge, CV_8UC1);
    cv::imshow("edge", edge);

    // OpenCV 函数，直接计算 scharr 算子卷积结果
    // void Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)
    cv::Mat img_scharrX_opencv, img_scharrY_opencv;
    cv::Scharr(image, img_scharrX_opencv, CV_32FC1, 1, 0, 1, 0, cv::BORDER_DEFAULT);
    cv::Scharr(image, img_scharrY_opencv, CV_32FC1, 0, 1, 1, 0, cv::BORDER_DEFAULT);

    cv::Mat img_scharr_opencv, img_scharrXX, img_scharrYY;
    cv::pow(img_scharrX_opencv, 2.0, img_scharrXX);
    cv::pow(img_scharrY_opencv, 2.0, img_scharrYY);
    cv::sqrt(img_scharrXX + img_scharrYY, img_scharr_opencv);
    // 数据类型转换，边缘强度的灰度级显示
    edge.convertTo(edge, CV_8UC1);
    cv::imshow("ScharrOpencv", img_scharr_opencv);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
