/**
 * @File    : local_threshold_otsu.cpp
 * @Brief   : 局部阈值分割; Otsu算法;
 * @Author  : Wei Li
 * @Date    : 2021-09-30
*/

#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat calcGrayHist(const cv::Mat &image)
{
    cv::Mat histogram = cv::Mat::zeros(cv::Size(256, 1), CV_32SC1);
    int rows = image.rows;
    int cols = image.cols;
    // 统计灰度等级数量
    for (size_t idx_row = 0; idx_row < rows; ++idx_row)
    {
        for (size_t idx_col = 0; idx_col < cols; ++idx_col)
        {
            int index = int(image.at<uchar>(idx_row, idx_col));
            histogram.at<int>(0, index) += 1;
        }
    }
    return histogram;
}

int otsuThreshold(const cv::Mat &image, cv::Mat &OtsuShresholdImage)
{
    cv::Mat histogram = calcGrayHist(image);
    cv::Mat normHist;
    histogram.convertTo(normHist, CV_32SC1, 1.0/(image.rows*image.cols), 0.0);
    // 计算累加直方图（零阶累积矩）和一阶累积矩
    cv::Mat zeroCumuMoment = cv::Mat::zeros(cv::Size(256, 1), CV_32SC1);
    cv::Mat oneCumuMoment = cv::Mat::zeros(cv::Size(256, 1), CV_32SC1);
    for (size_t i = 0; i < 256; ++i)
    {
        if (i == 0)
        {
            zeroCumuMoment.at<float>(0, i) = normHist.at<float>(0, i);
            oneCumuMoment.at<float>(0, i) = i * normHist.at<float>(0, i);
        }
        else
        {
            zeroCumuMoment.at<float>(0, i) = zeroCumuMoment.at<float>(0, i-1) + normHist.at<float>(0, i);
            oneCumuMoment.at<float>(0, i) = oneCumuMoment.at<float>(0, i-1) + i*normHist.at<float>(0, i);
        }
    }
    // 计算类间方差
    cv::Mat variance = cv::Mat::zeros(cv::Size(256, 1), CV_32SC1);
    // 总平均值
    float mean = oneCumuMoment.at<float>(0, 255);
    for (size_t i = 0; i < 255; ++i)
    {
        if (zeroCumuMoment.at<float>(0, i) == 0 || zeroCumuMoment.at<float>(0, i) == 1)
        {
            variance.at<float>(0, i) = 0;
        }
        else
        {
            float cofficient = zeroCumuMoment.at<float>(0, i) * (1.0 - zeroCumuMoment.at<float>(0, i));
            variance.at<float>(0, i) = cv::pow(mean*zeroCumuMoment.at<float>(0, i) - oneCumuMoment.at<float>(0, i), 2.0) / cofficient;
        }
    }
    // 找到阈值
    cv::Point maxLoc;
    cv::minMaxLoc(variance, nullptr, nullptr, nullptr, &maxLoc);
    int otsuThresh = maxLoc.x;
    cv::threshold(image, OtsuShresholdImage, otsuThresh, 255, cv::THRESH_BINARY);
    return otsuThresh;
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

        cv::Mat thresholdImg;
        int thresholdVal = otsuThreshold(image, thresholdImg);
        std::cout << "The threshold value is " << thresholdVal << std::endl;
        cv::imshow("ThresholdImage", thresholdImg);

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
