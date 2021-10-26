/**
 * @File    : discrete_convolution.cpp
 * @Brief   : 二维离散卷积(full，valid，same)；可分离卷积核以及其性质
 * @Author  : Wei Li
 * @Date    : 2021-09-17
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// -------------------------------
int main(int argc, char **argv)
{
    cv::Mat input_matrix = (cv::Mat_<float>(2, 2) << 1, 2, 3, 4);
    cv::Mat kernel_matrix = (cv::Mat_<float>(2, 2) << -1, -2, 2, 1);
    cv::Mat conv_same;

    // 1. 二维离散卷积 首先对 kernel 进行翻转180
    cv::Mat kernelFilp;
    cv::flip(kernel_matrix, kernelFilp, -1);
    // 然后利用 OpenCV 函数进行离散卷卷积操作(该函数只是相关计算，而不是卷积操作)
    cv::filter2D(input_matrix, conv_same, -1, kernelFilp, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);
    std::cout << "The result of same convolution: " << std::endl;
    std::cout << conv_same << std::endl;

    // 2. 可分离卷积
    cv::Mat convKernel = (cv::Mat_<float>(3, 3) << 1, 0, -1, 1, 0, -1, 1, 0, -1);
    cv::Mat convKernel_1 = (cv::Mat_<float>(3, 1) << 1, 1, 1);
    cv::Mat convKernel_2 = (cv::Mat_<float>(1, 3) << 1, 0, -1);
    cv::Mat convKernel_2_flip, separable_kernel;
    cv::flip(convKernel_2, convKernel_2_flip, -1);
    cv::filter2D(convKernel_1, separable_kernel, -1, convKernel_2_flip, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);
    // ???
    if (separable_kernel.data != convKernel.data)
    {
        std::cout << "The kernel of convolution is not Separable." << std::endl << std::endl;
    }

    // 3. 使用可分离卷积进行图像空间域滤波，减少计算量
    // 先进行垂直方向上的卷积，然后进行水平方向上的卷积
    cv::Mat inputMatrix = (cv::Mat_<float>(5, 5) << 1, 2, 3, 10, 12, 
                                                    32, 43, 12, 4, 190,
                                                    12, 234, 78, 0, 12,
                                                    43, 90, 32, 8, 90,
                                                    71, 12, 4, 98, 123);
    cv::Mat sameConv_h;
    cv::Mat sameConv_hv;
    // kernel 进行翻转180
    cv::Mat convKernel_2_filp, convKernel_1_filp;
    cv::flip(convKernel_2, convKernel_2_filp, -1);
    cv::flip(convKernel_1, convKernel_1_filp, -1);
    // 然后利用 OpenCV 函数进行离散卷卷积操作(该函数只是相关计算，而不是卷积操作)
    cv::filter2D(inputMatrix, sameConv_h, -1, convKernel_2_filp, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);
    cv::filter2D(sameConv_h, sameConv_hv, -1, convKernel_1_filp, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);
    std::cout << "The result of same convolution with Separable: " << std::endl;
    std::cout << sameConv_hv << std::endl;

    // 先进行水平方向上的卷积，然后进行垂直方向上的卷积
    cv::Mat sameConv_v;
    cv::Mat sameConv_vh;
    cv::filter2D(inputMatrix, sameConv_v, -1, convKernel_1_filp, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);
    cv::filter2D(sameConv_v, sameConv_vh, -1, convKernel_2_filp, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);
    std::cout << "The result of same convolution with Separable: " << std::endl;
    std::cout << sameConv_vh << std::endl;

    return 0;
}
