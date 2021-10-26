/**
 * @File    : guide_filter.cpp
 * @Brief   : 引导滤波(导向滤波); 快速导向滤波
 * @Author  : Wei Li
 * @Date    : 2021-09-25
*/

#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**导向滤波(引导滤波)
 * s 代表缩放系数 [0,1]
 * eps 表示正则化参数
 * r 表示窗口大小
 */
cv::Mat guideFilter(cv::Mat I, cv::Mat p, int r, float eps, float s)
{
    int rows = I.rows;
    int cols = I.cols;
    // 缩小图像
    cv::Mat small_I, small_p;
    cv::Size samllSize(int(std::round(s*cols)), int(std::round(s*rows)));
    cv::resize(I, small_I, samllSize, 0, 0, cv::INTER_CUBIC);
    cv::resize(p, small_p, samllSize, 0, 0, cv::INTER_CUBIC);

    // 缩放均值平滑窗口的半径
    int small_r = int(std::round(r*s));
    cv::Size winSize(2 * small_r + 1, 2 * small_r + 1);
    // 均值滤波平滑
    cv::Mat mean_small_I, mean_small_p;
    cv::boxFilter(small_I, mean_small_I, CV_64FC1, winSize);
    cv::boxFilter(small_p, mean_small_p, CV_64FC1, winSize);
    // small_I .* small_p 的均值平滑
    cv::Mat small_Ip = small_I.mul(small_p);
    cv::Mat mean_small_Ip;
    cv::boxFilter(small_Ip, mean_small_Ip, CV_64FC1, winSize);

    // 协方差
    cv::Mat cov_small_Ip = mean_small_Ip - mean_small_I.mul(mean_small_p);
    // 均值平滑
    cv::Mat mean_small_II;
    cv::boxFilter(small_I.mul(small_I), mean_small_II, CV_64FC1, winSize);

    // 方差
    cv::Mat var_small_I = mean_small_II - mean_small_I.mul(mean_small_I);
    cv::Mat small_a = cov_small_Ip / (var_small_I + eps);
    cv::Mat small_b = mean_small_p - small_a.mul(mean_small_I);

    // 对 small_a and small_b 进行均值平滑
    cv::Mat mean_small_a, mean_small_b;
    cv::boxFilter(small_a, mean_small_a, CV_64FC1, winSize);
    cv::boxFilter(small_b, mean_small_b, CV_64FC1, winSize);

    // 放大
    cv::Mat mean_a, mean_b;
    cv::resize(mean_small_a, mean_a, I.size(), 0, 0, cv::INTER_LINEAR);
    cv::resize(mean_small_b, mean_b, I.size(), 0, 0, cv::INTER_LINEAR);

    cv::Mat q = mean_a.mul(I) + mean_b;

    return q;
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

        // 灰度值归一化
        cv::Mat image_norm;
        image.convertTo(image_norm, CV_64FC1, 1.0/255, 0);
        // 导向滤波
        cv::Mat guide_filter_img = guideFilter(image_norm, image_norm, 7, 0.04, 0.3);
        cv::imshow("guide_filter_img", guide_filter_img);

        // 细节增强
        cv::Mat I_enchanced = (image_norm - guide_filter_img) * 5 + guide_filter_img;
        cv::normalize(I_enchanced, I_enchanced, 1, 0, cv::NORM_MINMAX, CV_32FC1);
        cv::imshow("I_enhanced", I_enchanced);

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
