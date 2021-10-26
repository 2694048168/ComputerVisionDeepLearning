/**
 * @File    : local_threshold_hist.cpp
 * @Brief   : 局部阈值分割；直方图技术
 * @Author  : Wei Li
 * @Date    : 2021-09-28
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

int threshTwoPeaks(const cv::Mat &image, cv::Mat &thresholdImg)
{
    cv::Mat histogram = calcGrayHist(image);
    // 找到灰度直方图最大峰值对应的灰度值
    cv::Point firstPeakLoc;
    cv::minMaxLoc(histogram, nullptr, nullptr, nullptr, &firstPeakLoc);
    int firstPeak = firstPeakLoc.x;
    // 寻找灰度直方图第二个峰值对应的灰度值
    cv::Mat measureDists = cv::Mat::zeros(cv::Size(256, 1), CV_32SC1);
    for (int k = 0; k < 256; ++k)
    {
        int hist_k = histogram.at<int>(0, k);
        measureDists.at<float>(0, k) = std::pow(float(k - firstPeak), 2)*hist_k;
    }
    cv::Point secondPeakLoc;
    cv::minMaxLoc(measureDists, nullptr, nullptr, nullptr, &secondPeakLoc);
    int secondPeak = secondPeakLoc.x;
    // 找到两个峰值之间最小值对应的灰度值，作为阈值
    cv::Point thresholdLoc;
    int thresholdVal = 0;
    // 第一个峰值在第二个峰值的左侧
    if (firstPeak < secondPeak)
    {
        cv::minMaxLoc(histogram.colRange(firstPeak, secondPeak), nullptr, nullptr, &thresholdLoc);
        thresholdVal = firstPeak + thresholdLoc.x + 1;
    }
    else // 第一个峰值在第二个峰值的右侧
    {
        cv::minMaxLoc(histogram.colRange(secondPeak, firstPeak), nullptr, nullptr, &thresholdLoc);
        thresholdVal =secondPeak + thresholdLoc.x + 1;
    }

    // 进行阈值分割
    cv::threshold(image, thresholdImg, thresholdVal, 255, cv::THRESH_BINARY);
    return thresholdVal;
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
        int thresholdVal = threshTwoPeaks(image, thresholdImg);
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
