/**
 * @File    : marr_hildreth.cpp
 * @Brief   : Marr-Hildreth 边缘检测 基于 高斯差分算子核高斯拉普拉斯算子
 * @Author  : Wei Li
 * @Date    : 2021-10-17
*/


#include <iostream>
#include <cmath>
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


void zero_cross_defalut(cv::InputArray _src, cv::OutputArray _dst)
{
    cv::Mat src = _src.getMat();
    // 判断位深
    CV_Assert(src.type() == CV_32FC1);
    _dst.create(src.size(), CV_8UC1);
    cv::Mat dst = _dst.getMat();

    int rows = src.rows;
    int cols = src.cols;
    // 交叉零点
    for (size_t row_idx = 1; row_idx < rows - 2; ++row_idx)
    {
        for (size_t col_idx = 1; col_idx < cols - 2; ++col_idx)
        {
            // 上下方向
            if (src.at<float>(row_idx - 1, col_idx)*src.at<float>(row_idx+1, col_idx) < 0)
            {
                dst.at<uchar>(row_idx, col_idx) = 255;
                continue;
            }
            
            // 左右方向
            if (src.at<float>(row_idx, col_idx-1)*src.at<float>(row_idx, col_idx+1) < 0)
            {
                dst.at<uchar>(row_idx, col_idx) = 255;
                continue;
            }
            
            // 左上/右下方向
            if (src.at<float>(row_idx-1, col_idx-1)*src.at<float>(row_idx+1, col_idx+1) < 0)
            {
                dst.at<uchar>(row_idx, col_idx) = 255;
                continue;
            }
            
            // 右上/左下方向
            if (src.at<float>(row_idx - 1, col_idx+1)*src.at<float>(row_idx+1, col_idx-1) < 0)
            {
                dst.at<uchar>(row_idx, col_idx) = 255;
                continue;
            }
        }
    }
}

void zero_cross_mean(cv::InputArray _src, cv::OutputArray _dst)
{
    cv::Mat src = _src.getMat();
    // 判断位深
    _dst.create(src.size(), CV_8UC1);
    cv::Mat dst = _dst.getMat();
    int rows = src.rows;
    int cols = src.cols;
    double minValue;
    double maxValue;

    // 存储四个方向的均值
    cv::Mat temp(1, 4, CV_32FC1);
    // 交叉零点
    for (size_t row_idx = 0+1; row_idx < rows - 1; ++row_idx)
    {
        for (size_t col_idx = 0+1; col_idx < cols; ++col_idx)
        {
            // 左上方向
            cv::Mat left_top(src, cv::Rect(col_idx - 1, row_idx - 1, 2, 2));
            temp.at<float>(0, 0) = cv::mean(left_top)[0];

            // 右上方向
            cv::Mat right_top(src, cv::Rect(col_idx, row_idx - 1, 2, 2));
            temp.at<float>(0, 1) = cv::mean(right_top)[0];

            // 左下方向
            cv::Mat left_bottom(src, cv::Rect(col_idx - 1, row_idx, 2, 2));
            temp.at<float>(0, 2) = cv::mean(left_bottom)[0];

            // 右下方向
            cv::Mat right_bottom(src, cv::Rect(col_idx, row_idx, 2, 2));
            temp.at<float>(0, 3) = cv::mean(right_bottom)[0];

            cv::minMaxLoc(temp, &minValue, &maxValue);
            // 最大值和最小值异号， 该位置为过零点
            if (minValue * maxValue < 0)
            {
                dst.at<uchar>(row_idx, col_idx) = 255;
            }
        }
    }
}

enum ZERO_CROSS_TYPE{
    ZERO_CROSS_DEFALUT,
    ZERO_CROSS_MEAN
};

void Marr_Hildreth_Image(cv::InputArray image, cv::OutputArray zeroCrossImage, int win, float sigma, ZERO_CROSS_TYPE zero_cross_type)
{
    // 高斯拉普拉斯
    cv::Mat imgLaplaceGauss;
    LaplaceGaussianOperator(image, imgLaplaceGauss, sigma, win);
    // 过零点
    switch (zero_cross_type)
    {
    case ZERO_CROSS_DEFALUT:
        zero_cross_defalut(imgLaplaceGauss, zeroCrossImage);
        break;
    
    case ZERO_CROSS_MEAN:
        zero_cross_defalut(imgLaplaceGauss, zeroCrossImage);
        break;
    
    default:
        std::cout << "Error: Not Implementation." << std::endl;
        break;
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

    // -------- Marr-Hildreth 边缘检测 --------
    cv::Mat edge;
    Marr_Hildreth_Image(image, edge, 7, 1, ZERO_CROSS_DEFALUT);
    cv::imshow("Marr-Hildreth_Edge", edge);
   
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
