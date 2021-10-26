/**
 * @File    : canny_operator.cpp
 * @Brief   : Canny 边缘检测算子 - 解决边缘梯度方向的信息问题和阈值处理问题
 * @Author  : Wei Li
 * @Date    : 2021-10-16
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

// 实现非极大值抑制的默认方式
cv::Mat non_maximum_suppression_default(cv::Mat dx, cv::Mat dy)
{
    // 使用平方和开方方式计算边缘强度
    cv::Mat edgeMag;
    cv::magnitude(dx, dy, edgeMag);

    int rows = dx.rows;
    int cols = dx.cols;
    // 边缘强度的非极大值抑制
    cv::Mat edgeMag_nonMaxSup = cv::Mat::zeros(dx.size(), dx.type());
    for (size_t row_idx = 1; row_idx < rows - 1; ++row_idx)
    {
        for (size_t col_idx = 1; col_idx < cols; ++col_idx)
        {
            float x = dx.at<float>(row_idx, col_idx);
            float y = dy.at<float>(row_idx, col_idx);
            // 梯度方向
            float angle = std::atan2(y, x) / CV_PI * 180;
            // 当前位置的边缘强度
            float mag = edgeMag.at<float>(row_idx, col_idx);

            // 左/右 方向比较
            if (cv::abs(angle) < 22.5 || cv::abs(angle) > 157.5)
            {
                float left = edgeMag.at<float>(row_idx, col_idx - 1);
                float right = edgeMag.at<float>(row_idx, col_idx + 1);
                if (mag > left && mag < right)
                {
                    edgeMag_nonMaxSup.at<float>(row_idx, col_idx) = mag;
                }
            }

            // 左上/右下 方向比较
            if ((angle >= 22.5 && angle < 67.5) || (angle < -112.5 && angle >= 157.5))
            {
                float leftTop = edgeMag.at<float>(row_idx - 1, col_idx - 1);
                float rightBottom = edgeMag.at<float>(row_idx + 1, col_idx + 1);
                if (mag > leftTop && mag < rightBottom)
                {
                    edgeMag_nonMaxSup.at<float>(row_idx, col_idx) = mag;
                }
            }

            // 上/下 方向比较
            if ((angle >= 67.5 && angle <= 112.5) || (angle < -112.5 && angle >= -67.5))
            {
                float top = edgeMag.at<float>(row_idx - 1, col_idx);
                float bottom = edgeMag.at<float>(row_idx + 1, col_idx);
                if (mag > top && mag < bottom)
                {
                    edgeMag_nonMaxSup.at<float>(row_idx, col_idx) = mag;
                }
            }

            // 右上/左下 方向比较
            if ((angle > 112.5 && angle <= 157.5) || (angle > -67.5 && angle <= -22.5))
            {
                float rightTop = edgeMag.at<float>(row_idx - 1, col_idx + 1);
                float leftBottom = edgeMag.at<float>(row_idx + 1, col_idx - 1);
                if (mag > rightTop && mag < leftBottom)
                {
                    edgeMag_nonMaxSup.at<float>(row_idx, col_idx) = mag;
                }
            }
        }
    }
    return edgeMag_nonMaxSup;
}

// 实现非极大值抑制的插值方式
cv::Mat non_maximum_suppression_interpolation(cv::Mat dx, cv::Mat dy)
{
    // 使用平方和开方方式计算边缘强度
    cv::Mat edgeMag;
    cv::magnitude(dx, dy, edgeMag);

    int rows = dx.rows;
    int cols = dx.cols;
    // 边缘强度的非极大值抑制
    cv::Mat edgeMag_nonMaxSup = cv::Mat::zeros(dx.size(), dx.type());
    for (size_t row_idx = 1; row_idx < rows - 1; ++row_idx)
    {
        for (size_t col_idx = 1; col_idx < cols; ++col_idx)
        {
            float x = dx.at<float>(row_idx, col_idx);
            float y = dy.at<float>(row_idx, col_idx);
            if (x == 0 && y == 0)
            {
                continue;
            }
            
            // 梯度方向
            float angle = std::atan2(y, x) / CV_PI * 180;
            // 邻域内 8 个方向上的边缘强度
            float leftTop = edgeMag.at<float>(row_idx - 1, col_idx - 1);
            float top = edgeMag.at<float>(row_idx - 1, col_idx);
            float rightBottom = edgeMag.at<float>(row_idx + 1, col_idx + 1);
            float right = edgeMag.at<float>(row_idx, col_idx + 1);
            float rightTop = edgeMag.at<float>(row_idx - 1, col_idx + 1);
            float leftBottom = edgeMag.at<float>(row_idx + 1, col_idx - 1);
            float bottom = edgeMag.at<float>(row_idx + 1, col_idx);
            float left = edgeMag.at<float>(row_idx, col_idx - 1);
            // 当前位置的边缘强度
            float mag = edgeMag.at<float>(row_idx, col_idx);

            // 左上方和上方的插值、右下方和下方的插值
            if ((angle > 45 && angle <= 90) || (angle > -135 && angle <= -90))
            {
                float ratio = x / y;
                float top = edgeMag.at<float>(row_idx - 1, col_idx);
                // 插值
                float leftTop_top = ratio * leftTop + (1 - ratio) * top;
                float rightBottom_bottom = ratio * rightBottom + (1 - ratio) * bottom;
                if (mag > leftTop_top && mag > rightBottom_bottom)
                {
                    edgeMag_nonMaxSup.at<float>(row_idx, col_idx) = mag;
                }
            }
            
            // 右上方和上方的插值、左下方和下方的插值
            if ((angle > 90 && angle <= 135) || (angle > -90 && angle <= -45))
            {
                float ratio = cv::abs(x / y);
                // 插值
                float rightTop_top = ratio * rightTop + (1 - ratio) * top;
                float leftBottom_bottom = ratio * leftBottom + (1 - ratio) * bottom;
                if (mag > rightTop_top && mag > leftBottom_bottom)
                {
                    edgeMag_nonMaxSup.at<float>(row_idx, col_idx) = mag;
                }
            }
            
            // 左上方和左方的插值、右下方和右方的插值
            if ((angle >= 0 && angle <= 45) || (angle > -180 && angle <= -135))
            {
                float ratio = y / x;
                // 插值
                float rightBottom_right = ratio * rightBottom + (1 - ratio) * right;
                float leftTop_left = ratio * leftTop + (1 - ratio) * left;
                if (mag > rightBottom_right && mag > leftTop_left)
                {
                    edgeMag_nonMaxSup.at<float>(row_idx, col_idx) = mag;
                }
            }
            
            // 右上方和右方的插值、左下方和左方的插值
            if ((angle >= 135 && angle <= 180) || (angle > -45 && angle <= 0))
            {
                float ratio = cv::abs(y / x);
                // 插值
                float rightTop_right = ratio * rightTop + (1 - ratio) * right;
                float leftBottom_left = ratio * leftBottom + (1 - ratio) * left;
                if (mag > rightTop_right && mag > leftBottom_left)
                {
                    edgeMag_nonMaxSup.at<float>(row_idx, col_idx) = mag;
                }
            }
        }
    }
    return edgeMag_nonMaxSup;
}

// 确定一个点的坐标是否在图像范围内
bool checkInRange(int row_idx, int col_idx, int rows, int cols)
{
    if (row_idx >= 0 && row_idx < rows && col_idx >= 0 && col_idx < cols)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// 从确定边缘出发，延长边缘
void trace(cv::Mat edgeMag_nonMaxSup, cv::Mat &edge, float lowerThresh, int row_idx, int col_idx, int rows, int cols)
{
    if (edge.at<uchar>(row_idx, col_idx) == 0)
    {
        edge.at<uchar>(row_idx, col_idx) = 255;
        for (size_t i = -1; i <= 1; ++i)
        {
            for (size_t j = -1; i <= 1; ++j)
            {
                float mag = edgeMag_nonMaxSup.at<float>(row_idx+i, col_idx+j);
                if (checkInRange(row_idx+i, col_idx+j, rows, cols) && mag >= lowerThresh)
                {
                    trace(edgeMag_nonMaxSup, edge, lowerThresh, row_idx+i, col_idx+j, rows, cols);
                }
            }
        }
    }
}

// 双阈值的滞后阈值处理
cv::Mat hysteresisThreshold(cv::Mat edgeMag_nonMaxSup, float lowerThresh, float upperThresh)
{
    int rows = edgeMag_nonMaxSup.rows;
    int cols = edgeMag_nonMaxSup.cols;
    // 最终边缘输出图
    cv::Mat edge = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);

    // 滞后阈值处理
    for (size_t row_idx = 1; row_idx < rows - 1; ++row_idx)
    {
        for (size_t col_idx = 1; col_idx < cols; ++col_idx)
        {
            float mag = edgeMag_nonMaxSup.at<float>(row_idx, col_idx);
            // 大于高阈值的点，作为边缘点，并以该点为起始点延长边缘
            if (mag >= upperThresh)
            {
                trace(edgeMag_nonMaxSup, edge, lowerThresh, row_idx, col_idx, rows, cols);
            }
            // 小于低阈值的点直接删掉
            if (mag < lowerThresh)
            {
                edge.at<uchar>(row_idx, col_idx) = 0;
            }
        }
    }
    return edge;
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

    // -------- Canny 边缘检测 --------
    // step 1, Sobel kernel convolution
    cv::Mat img_sobe1_x = SobelOperator(image, 1, 0, 3, cv::BORDER_DEFAULT);
    cv::Mat img_sobe1_y = SobelOperator(image, 0, 1, 3, cv::BORDER_DEFAULT);

    // step 2, Edge strength
    cv::Mat edge, img_sobe1_xx, img_sobe1_yy;
    cv::pow(img_sobe1_x, 2.0, img_sobe1_xx);
    cv::pow(img_sobe1_y, 2.0, img_sobe1_yy);
    cv::sqrt(img_sobe1_xx + img_sobe1_yy, edge);
    // // 数据类型转换，边缘强度的灰度级显示
    // edge.convertTo(edge, CV_8UC1);
    // cv::imshow("edgeStrength", edge);

    // step 3, Non-maximum suppression
    cv::Mat edgeMag_nonMaxSup = non_maximum_suppression_default(img_sobe1_x, img_sobe1_y);
    // // 数据类型转换，边缘强度的灰度级显示
    // edgeMag_nonMaxSup.convertTo(edgeMag_nonMaxSup, CV_8UC1);
    // cv::imshow("edgeNonMaxSup", edgeMag_nonMaxSup);

    // step 4, Dual threshold hysteresis precessing
    edge = hysteresisThreshold(edgeMag_nonMaxSup, 20, 150);
    cv::imshow("edge", edge);

    // OpenCV 函数，直接计算 Scanny 算子
    cv::Mat edgesCanny;
    cv::Canny(image, edgesCanny, 20, 150, 3, true);
    cv::imshow("EdgeCanny", edgesCanny);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
