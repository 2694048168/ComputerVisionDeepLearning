/**
 * @File    : laplace_gaussian_operator.cpp
 * @Brief   : 高斯拉普拉斯算子 —— 先二维高斯平滑处理，后进行拉普拉斯微分算子 —— 可分离高斯拉普拉斯卷积核
            拉普拉斯算子对噪声很敏感，使用首先应对图像进行高斯平滑，然后再与拉普拉斯算子卷积，最后得到二值化边缘图。
 * @Author  : Wei Li
 * @Date    : 2021-10-17
*/

#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


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

void getSepLoGKernel(float sigma, int length, cv::Mat &kernelX, cv::Mat &kernelY)
{
    kernelX.create(cv::Size(length, 1), CV_32FC1);
    kernelY.create(cv::Size(1, length), CV_32FC1);
    int center = (length - 1) / 2;
    double sigma2 = std::pow(sigma, 2.0);

    // 构建可分离的高斯拉普拉斯算子
    for (size_t c = 0; c < length; ++c)
    {
        float norm2 = std::pow(c - center, 2.0);
        kernelY.at<float>(c, 0) = std::exp(-norm2 / (2 * sigma2));
        kernelX.at<float>(0, c) = (norm2 / sigma2 - 1.0)*kernelY.at<float>(c, 0);
    }
}

void LaplaceGaussianOperator(cv::InputArray image, cv::OutputArray LoGConv, float sigma, int winSize)
{
    cv::Mat kernelX, kernelY;
    getSepLoGKernel(sigma, winSize, kernelX, kernelY);

    // 先进行水平卷积，再进行垂直卷积
    cv::Mat convXY;
    sepConv2D_X_Y(image, convXY, CV_32FC1, kernelX, kernelY);

    // 卷积核转置
    cv::Mat kernelX_T = kernelX.t();
    cv::Mat kernelY_T = kernelY.t();

    // 先进行垂直卷积，再进行水平卷积
    cv::Mat convYX;
    sepConv2D_Y_X(image, convYX, CV_32FC1, kernelX_T, kernelY_T);

    // 计算两个卷积结果的和，获取高斯拉普拉斯卷积结果
    cv::add(convXY, convYX, LoGConv);
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

    // -------- Laplace of Gaussian Operator --------
    float sigma = 1;
    int winSize = 7;
    cv::Mat LoGConv;
    LaplaceGaussianOperator(image, LoGConv, sigma, winSize);

    // 阈值化处理
    cv::Mat edgeLapace;
    cv::threshold(LoGConv, edgeLapace, 0, 255, cv::THRESH_BINARY);
    cv::imshow("BinaryImage", edgeLapace);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
