/**
 * @File    : gauss_kernel.cpp
 * @Brief   : 计算高斯卷积算子；高斯卷积核的可分离；高斯卷积核进行图像平滑(模糊)
 * @Author  : Wei Li
 * @Date    : 2021-09-17
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat sepConv2D_Y_X(const cv::Mat &image, cv::Mat gaussKernel_y, cv::Mat gaussKernel_x)
{
    // 可分离高斯卷积
    cv::Mat sep_gauss_img_y, sep_gauss_img;
    // 然后利用 OpenCV 函数进行离散卷卷积操作(该函数只是相关计算，而不是卷积操作)
    cv::filter2D(image, sep_gauss_img_y, -1, gaussKernel_y, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);
    cv::filter2D(sep_gauss_img_y, sep_gauss_img, -1, gaussKernel_x, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);

    return sep_gauss_img;
}

cv::Mat gaussBlur(const cv::Mat &image, cv::Size winSize, float sigma,
                  int ddepth = CV_64F, cv::Point anchor = cv::Point(-1, -1), int borderType = cv::BORDER_DEFAULT)
{
    // 高斯卷积核为奇数
    CV_Assert(winSize.width % 2 == 1 && winSize.height % 2 == 1);
    // 垂直方向高斯卷积核
    cv::Mat gaussKernel_y = cv::getGaussianKernel(sigma, winSize.height, CV_64F);
    // 水平方向 and transpose
    cv::Mat gaussKernel_x = cv::getGaussianKernel(sigma, winSize.width, CV_64F);
    gaussKernel_x = gaussKernel_x.t(); // 转置
    // 可分离的高斯卷积核,自定义函数
    cv::Mat blurImgae = sepConv2D_Y_X(image, gaussKernel_y, gaussKernel_x);

    return blurImgae;
}

// -------------------------------
int main(int argc, char **argv)
{
    if (argc > 1)
    {
        cv::Mat image = cv::imread(argv[1], 0);
        if (!image.data)
        {
            std::cout << "Error: reading image unsuccesfully." << std::endl;
            return -1;
        }
        cv::imshow("OriginImage", image);

        // 1. 自定义高斯卷积
        cv::Mat first_gauss_blur_img = gaussBlur(image, cv::Size(7, 7), 1.5);
        cv::imshow("firstGaussBlurImg", first_gauss_blur_img);

        // 2. OpenCV 函数进行高斯卷积
        cv::Mat second_gauss_blur_img;
        cv::GaussianBlur(image, second_gauss_blur_img, cv::Size(7, 7), 1.5, 1.5);
        cv::imshow("secondGaussBlurImg", second_gauss_blur_img);

        cv::waitKey(0);
        cv::destroyAllWindows();
        return 0;
    }
    else
    {
        std::cout << "Usage: OpenCV gauss convolution image." << std::endl;
        return -1;
    }
}
