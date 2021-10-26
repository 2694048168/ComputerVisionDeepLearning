/**
 * @File    : integral_mean.cpp
 * @Brief   : 图像积分; 快速均值滤波(平滑)
 * @Author  : Wei Li
 * @Date    : 2021-09-23
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat fastMeanBlur(cv::Mat img, cv::Size winSize, int borderType, cv::Scalar value = cv::Scalar())
{
    // 滤波核宽和高为奇数
    int hei = winSize.height;
    int wid = winSize.width;
    CV_Assert(hei % 2 == 1 && wid % 2 == 1);
    // 窗口半径
    int hh = (hei - 1) / 2;
    int ww = (wid - 1) / 2;
    // 窗口面积
    float area = float(hei * wid);
    // 边界填充
    cv::Mat paddImg;
    cv::copyMakeBorder(img, paddImg, hh, hh, ww, ww, borderType, value);
    // 图像积分
    cv::Mat inte;
    cv::integral(paddImg, inte, CV_32FC1);
    // 输入图像的高和宽
    int rows = img.rows;
    int cols = img.cols;
    int r = 0, c = 0;
    cv::Mat meanImg = cv::Mat::zeros(img.size(), CV_32FC1);
    for (size_t h = hh; h < hh + rows; ++h)
    {
        for (size_t w = ww; w < ww + cols; ++w)
        {
            float bottomRight = inte.at<float>(h + hh + 1, w + ww + 1);
            float topLeft = inte.at<float>(h - hh, w - ww);
            float topRight = inte.at<float>(h + hh + 1, w - ww);
            float bottomLeft = inte.at<float>(h - hh, w + ww + 1);
            meanImg.at<float>(r, c) = (bottomRight + topLeft - topRight - bottomLeft) / area;
            ++c;
        }
        ++r;
        c = 0;
    }

    return meanImg;
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

        // 1. fastMeanBlur
        cv::Mat fastMeanBlurImg = fastMeanBlur(image, cv::Size(5, 5), 4);
        cv::imshow("fastMeanBlurImg", fastMeanBlurImg);

        // 2. OpenCV 函数进行
        // 对于快速均值平滑，OpenCV提供了boxFilter和 blur两个函数来实现该功能，
        // 而且这两个函数均可以处理多通道图像矩阵，本质上是对图像的每一个通道分别进行均值平滑
        cv::Mat second_mean_blur_img;
        cv::blur(image, second_mean_blur_img, cv::Size(5, 5), cv::Point(-1, -1));
        cv::imshow("second_mean_blur_img", second_mean_blur_img);

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
