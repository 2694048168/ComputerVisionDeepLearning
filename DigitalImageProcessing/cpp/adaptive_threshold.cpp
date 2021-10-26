/**
 * @File    : adaptive_threshold.cpp
 * @Brief   : 局部阈值分割; 自适应阈值分割;
 * @Author  : Wei Li
 * @Date    : 2021-09-30
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

enum METHOD {MEAN, GAUSS, MEDIAN};

cv::Mat adaptiveThresh(cv::Mat image, int radius, float ratio, METHOD method=MEAN)
{
    // 1. 对图像进行平滑处理
    cv::Mat smooth;
    switch (method)
    {
    case MEAN:
        cv::boxFilter(image, smooth, CV_32FC1, cv::Size(2*radius+1, 2*radius+1));
        break;
    case GAUSS:
        cv::GaussianBlur(image, smooth, cv::Size(2*radius+1, 2*radius+1), 0, 0);
        break;
    case MEDIAN:
        cv::medianBlur(image, smooth, 2*radius+1);
        break;
    default:
        break;
    }
    // 2. 平滑结果乘以比例系数，然后图像矩阵与其做残差
    image.convertTo(image, CV_32FC1);
    smooth.convertTo(smooth, CV_32FC1);
    cv::Mat diff = image - (1.0 - ratio)*smooth;
    // 3. 阈值处理，当差值大于或者等于 0 时， 输出值为 255；反之输出值为 0
    cv::Mat out_img = cv::Mat::zeros(diff.size(), CV_8UC1);
    for (int r = 0; r < out_img.rows; ++r)
    {
        for (int c = 0; c < out_img.cols; ++c)
        {
            if (diff.at<float>(r, c) >= 0)
            {
                out_img.at<uchar>(r, c) = 255;
            }
        }
    }
    return out_img;
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

        cv::Mat thresholdImg = adaptiveThresh(image, 5, 0.15);
        cv::imshow("ThresholdImage", thresholdImg);

        // 利用 opencv 函数实现
        cv::Mat adaptiveImg;
        cv::adaptiveThreshold(image, adaptiveImg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 0.15);
        cv::imshow("AdaptiveThresholdImage", adaptiveImg);

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
