/**
 * @File    : laplace_diff_gaussian.cpp
 * @Brief   : 高斯差分边缘检测(接近高斯拉普拉斯算子) —— 计算量减少
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

/**函数 gaussConv 实现图像矩阵与分离高斯核的卷积，
 * 其中参数 image 代表输入的图像矩阵;
 * sigma 代表标准差;
 * 经常使用的是高和宽相同的高斯核，所以令 s 代表高斯核的高和宽，即高斯核的尺寸为 sxs 且 s 为奇数。
 * 返回值是数据类型为 CV_32F 的卷积结果，当然可以换成其他数据类型。
 * 分离卷积使用的仍然是函数 sepConv2D_X_Y。
 */
void gaussConv(const cv::Mat image, cv::Mat &imgGuass, float sigma, int s)
{
    // 构建水平方向的非归一化高斯核
    cv::Mat x_kernel = cv::Mat::zeros(1, s, CV_32FC1);
    int anchor = (s - 1) / 2;
    float sigma2 = std::pow(sigma, 2.0);
    for (size_t i = 0; i < s; ++i)
    {
        float norm2 = std::pow(float(i - anchor), 2.0);
        x_kernel.at<float>(0, i) = std::exp(-norm2 / (2 * sigma2));
    }

    // 将 x_kernel 转置，获取垂直方向的卷积核
    cv::Mat y_kernel = x_kernel.t();

    // 利用可分离卷积核完成卷积计算
    sepConv2D_X_Y(image, imgGuass, CV_32F, x_kernel, y_kernel);
    imgGuass.convertTo(imgGuass, CV_32F, 1.0 / sigma2);
}

void DiffGuass(const cv::Mat image, cv::Mat &imgDiffGauss, float sigma, int s, float k=1.1)
{
    // 标准差 sigma 的非归一化高斯核卷积
    cv::Mat img_gauss_1;
    gaussConv(image, img_gauss_1, sigma, s);

    // 标准差 k*sigma 的非归一化高斯卷积
    cv::Mat img_gauss_k;
    gaussConv(image, img_gauss_k, k*sigma, s);

    // 两个高斯卷积结果做差
    imgDiffGauss = img_gauss_k - img_gauss_1;
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

    // -------- Difference of Gaussian Operator --------
    float sigma = 2;
    int winSize = 13;
    float k = 1.05;
    cv::Mat imgDiffGauss;
    DiffGuass(image, imgDiffGauss, sigma, winSize, k);

    // 阈值化处理
    cv::Mat imgDiffGaussBinary;
    cv::threshold(imgDiffGauss, imgDiffGaussBinary, 0, 255, cv::THRESH_BINARY);
    cv::imshow("DiffGuassImage", imgDiffGaussBinary);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
