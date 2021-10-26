/**
 * @File    : hough_transform.cpp
 * @Brief   : 霍夫变换 (Hough Transform) 进行二值图像的直线检测
 * @Author  : Wei Li
 * @Date    : 2021-10-18
*/


#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


std::map<std::vector<int>, std::vector<cv::Point>> HoughTransformLine(cv::Mat image, cv::Mat &accumulator, float stepTheta=1, float stepRho=1)
{
    int rows = image.rows;
    int cols = image.cols;
    // 可能出现的最大垂线的长度
    int L = std::round(std::sqrt(std::pow(rows - 1, 2.0) + std::pow(cols - 1, 2.0))) + 1;
    // 初始化投票器
    int numTheta = int(180.0 / stepTheta);
    int numRho = int(2 * L / stepRho + 1);
    accumulator = cv::Mat::zeros(cv::Size(numTheta, numRho), CV_32SC1);

    // 初始化类 map ，存储共线的点
    std::map<std::vector<int>, std::vector<cv::Point>> lines;
    for (size_t i = 0; i < numRho; ++i)
    {
        for (size_t j = 0; i < numTheta; ++j)
        {
            lines.insert(std::make_pair(std::vector<int>(j, i), std::vector<cv::Point>()));
        }
    }
    // 投票计数
    for (size_t y = 0; y < rows; ++y)
    {
        for (size_t x = 0; x < cols; ++x)
        {
            if (image.at<uchar>(cv::Point(x, y)) == 255)
            {
                for (size_t m = 0; m < numTheta; ++m)
                {
                    // 对每一个角度，计算对应的 rho 值 
                    float rho1 = x * std::cos(stepTheta * m / 180.0 * CV_PI);
                    float rho2 = y * std::sin(stepTheta * m / 180.0 * CV_PI);
                    float rho = rho1 + rho2;
                    // 计算投票到哪一个区域
                    int n = int(std::round(rho + L) / stepRho);
                    // 累加 1
                    accumulator.at<int>(n, m) += 1;
                    // 记录该点
                    lines.at(std::vector<int>(m, n)).push_back(cv::Point(x, y));
                }
            }
        }
    }
    return lines;
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

        // 边缘二值图
        cv::Mat edge;
        cv::Canny(image, edge, 50, 200);

        // ------------ Hough Transform Line ------------
        cv::Mat accu;
        std::map<std::vector<int>, std::vector<cv::Point>> lines;
        lines = HoughTransformLine(edge, accu);

        // 投票器的灰度级可视化
        double maxValue;
        cv::minMaxLoc(accu, nullptr, &maxValue, nullptr, nullptr);
        cv::Mat grayAccu;
        accu.convertTo(grayAccu, CV_32FC1, 1.0 / maxValue);
        cv::imshow("AccumulatorGray", grayAccu);
        // 绘制投票数量大于某一阈值的直线
        int vote = 150;
        for (size_t r = 1; r < accu.rows - 1; ++r)
        {
            for (size_t c = 1; c < accu.cols - 1; ++c)
            {
                int current = accu.at<int>(r, c);
                if (current > vote)
                {
                    int left_top = accu.at<int>(r - 1, c - 1);
                    int top = accu.at<int>(r - 1, c);
                    int right_top = accu.at<int>(r - 1, c + 1);
                    int left = accu.at<int>(r, c - 1);
                    int right = accu.at<int>(r, c + 1);
                    int left_bottom = accu.at<int>(r + 1, c - 1);
                    int bottom = accu.at<int>(r - 1, c);
                    int right_bottom = accu.at<int>(r + 1, c + 1);
                    // 判断该位置是不是局部最大值
                    if (current > left && current > top && current > right_top && current > left && current < right && current > left_bottom && current > bottom && current > right_bottom)
                    {
                        std::vector<cv::Point> line = lines.at(std::vector<int>(c, r));
                        int s = line.size();
                        cv::line(image, line.at(0), line.at(s - 1), cv::Scalar(255), 2);
                    }
                }
            }
        }

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
