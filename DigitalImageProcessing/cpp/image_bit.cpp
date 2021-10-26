/**
 * @File    : image_bit.cpp
 * @Brief   : 二值图的逻辑运算
 * @Author  : Wei Li
 * @Date    : 2021-09-30
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// -------------------------------
int main(int argc, char **argv)
{
    cv::Mat src1 = cv::Mat::zeros(cv::Size(100, 100), CV_8UC1);
    cv::rectangle(src1, cv::Rect(23, 23, 50, 50), cv::Scalar(255), cv::FILLED);
    cv::imshow("src1", src1);

    cv::Mat src2 = cv::Mat::zeros(cv::Size(100, 100), CV_8UC1);
    cv::circle(src1, cv::Point(75, 50), 25, cv::Scalar(255), cv::FILLED);
    cv::imshow("src2", src2);

    // 与运算
    cv::Mat dst_and;
    cv::bitwise_and(src1, src2, dst_and);
    cv::imshow("and_bit", dst_and);

    // 或运算
    cv::Mat dst_or;
    cv::bitwise_or(src1, src2, dst_or);
    cv::imshow("or_bit", dst_or);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
