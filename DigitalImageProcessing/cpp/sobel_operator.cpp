/**
 * @File    : sobel_operator.cpp
 * @Brief   : Sobel 算子进行边缘检测  —— 可分离卷积核
 * @Author  : Wei Li
 * @Date    : 2021-10-14
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// 可以将这一系列的函数操作封装为一个类
// 卷积运算 OpenCV with CPlusPlus
void conv2D(cv::InputArray src, cv::InputArray kernel, cv::OutputArray dst, int ddepth, cv::Point anchor=cv::Point(-1, -1), int borderType=cv::BORDER_DEFAULT)
{
    // 1. 卷积核逆时针翻转 180 度
    cv::Mat kernelFlip;
    cv::flip(kernel, kernelFlip, -1);
    // 利用计算自相关函数完成卷积运算
    cv::filter2D(src, dst, ddepth, kernelFlip, anchor, 0.0, borderType);
}

// 可分离卷积运算
// 可分离离散二维卷积运算，先进行垂直方向卷积，再进行水平方向卷积
void sepConv2D_Y_X(cv::InputArray src, cv::OutputArray src_kerY_kerX, int ddepth, cv::InputArray kernelY, cv::InputArray kernelX, cv::Point anchor=cv::Point(-1, -1), int borderType=cv::BORDER_DEFAULT)
{
    cv::Mat src_kerY;
    conv2D(src, kernelY, src_kerY, ddepth, anchor, borderType);
    conv2D(src_kerY, kernelX, src_kerY_kerX, ddepth, anchor, borderType);
}

// 可分离离散二维卷积运算，先进行水平方向卷积，再进行垂直方向卷积
void sepConv2D_X_Y(cv::InputArray src, cv::OutputArray src_kerX_kerY, int ddepth, cv::InputArray kernelX, cv::InputArray kernelY, cv::Point anchor=cv::Point(-1, -1), int borderType=cv::BORDER_DEFAULT)
{
    cv::Mat src_kerX;
    conv2D(src, kernelX, src_kerX, ddepth, anchor, borderType);
    conv2D(src_kerX, kernelY, src_kerX_kerY, ddepth, anchor, borderType);
}


// Factorial 函数实现阶乘
int Factorial(int n)
{
    int fac = 1;
    if (n == 0)
    {
        return fac;
    }
    for (int i = 1; i <= n; ++i)
    {
        fac *= i;
    }

    return fac;
}

// getPascalSmooth 函数实现获取 soble 平滑算子
cv::Mat getPascalSmooth(int n)
{
    cv::Mat pascalSmooth = cv::Mat::zeros(cv::Size(n, 1), CV_32FC1);
    for (int i = 0; i < n; ++i)
    {
        pascalSmooth.at<float>(0, i) = Factorial(n - 1) / (Factorial(i) * Factorial(n - 1 - i));
    }

    return pascalSmooth;
}

// getPascalDiff 函数实现获取 soble 差分算子
cv::Mat getPascalDiff(int n)
{
    cv::Mat pascalDiff = cv::Mat::zeros(cv::Size(n, 1), CV_32FC1);
    cv::Mat pascalSmooth_previous = getPascalSmooth(n - 1);
    for (int i = 0; i < n; ++i)
    {
        if (i == 0) // 恒等于 1
        {
            pascalDiff.at<float>(0, i) = 1;
        }
        else if (i == n - 1) // 恒等于 -1
        {
            pascalDiff.at<float>(0, i) = -1;
        }
        else
        {
            pascalDiff.at<float>(0, i) = pascalSmooth_previous.at<float>(0, i) - pascalSmooth_previous.at<float>(0, i - 1);
        }
    }

    return pascalDiff;
}

// SobelOperator 函数实现图像核 soble 算子的卷积操作
// 当参数 x_flag !=0 时，返回图像矩阵与 sobel_x 核的卷积 ;
// 当 x_flag = 0且 y_flag !=0 时，返回图像矩阵与 sobel_y 核的卷积。
cv::Mat SobelOperator(cv::Mat image, int x_flag, int y_flag, int winSize, int borderType)
{
    // sobel 卷积核的大小为大于 3 的奇数
    CV_Assert(winSize >= 3 && winSize % 2 == 1);
    // 平滑系数
    cv::Mat pascalSmooth = getPascalSmooth(winSize);
    // 差分系数
    cv::Mat pascalDiff = getPascalDiff(winSize);
    cv::Mat img_sobel_conv;

    // ---- x_flag != 0, image and soble_x kernel convolution ----Horizontal
    if (x_flag != 0)
    {
        // 可分离卷积性质：先进行一维垂直方向的平滑，再进行一维水平方向的差分
        sepConv2D_Y_X(image, img_sobel_conv, CV_32FC1, pascalSmooth.t(), pascalDiff, cv::Point(-1, -1), borderType);
    }

    // ---- x_flag != 0, image and soble_x kernel convolution ----Vertical
    if (x_flag == 0 && y_flag != 0)
    {
        // 可分离卷积性质：先进行一维水平方向的平滑，再进行一维垂直方向的差分
        sepConv2D_X_Y(image, img_sobel_conv, CV_32FC1, pascalSmooth, pascalDiff.t(), cv::Point(-1, -1), borderType);
    }

    return img_sobel_conv;
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

    // 原始图像和 soble kernerl-x 卷积
    cv::Mat img_sobel_x = SobelOperator(image, 1, 0, 3, cv::BORDER_DEFAULT);

    // 原始图像和 soble kernerl-y 卷积
    cv::Mat img_sobel_y = SobelOperator(image, 0, 1, 3, cv::BORDER_DEFAULT);

    // 两个卷积结果的灰度级显示
    cv::Mat abs_img_sobel_x, abs_img_sobel_y;
    cv::convertScaleAbs(img_sobel_x, abs_img_sobel_x, 1, 0);
    cv::convertScaleAbs(img_sobel_y, abs_img_sobel_y, 1, 0);
    cv::imshow("horizontal_edge", abs_img_sobel_x);
    cv::imshow("vertical_edge", abs_img_sobel_y);

    // 根据 soble 两个卷积结果，获取最终图像的边缘强度
    cv::Mat edge;
    cv::Mat img_sobel_x_x, img_sobel_y_y;
    cv::pow(img_sobel_x, 2.0, img_sobel_x_x);
    cv::pow(img_sobel_y, 2.0, img_sobel_y_y);
    cv::sqrt(img_sobel_x_x + img_sobel_y_y, edge);
    // 数据类型转换，边缘强度的灰度级显示
    edge.convertTo(edge, CV_8UC1);
    cv::imshow("edge", edge);

    // OpenCV 函数，直接计算 sobel 算子卷积结果
    // void Sobel(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType)

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
