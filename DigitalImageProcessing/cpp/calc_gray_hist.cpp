/**
 * @File    : calc_gray_hist.cpp
 * @Brief   : 计算灰度直方图(归一化直方图，概率直方图)
 * @Author  : Wei Li
 * @Date    : 2021-09-15
*/

#include <iostream>
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

// -------------------------------
int main(int argc, char** argv)
{
    // tips for coding, 实现矩阵的平铺 repeat(tile() function in Numpy)
    if (argc > 1)
    {
        cv::Mat image = cv::imread(argv[1], 0);
        if (!image.data)
        {
            std::cout << "Error: reading image unsuccesfully." << std::endl;
            return -1;
        }
        cv::imshow("OriginImage", image);

        // 计算图像的灰度直方图
        cv::Mat histogram_img = calcGrayHist(image);
        std::cout << "The histogram of gray image: \n" << histogram_img << std::endl;
        // 可视化灰度直方图

        cv::waitKey(0);
        cv::destroyAllWindows();
        return 0;
    }
    else
    {
        std::cout << "Usage: OpenCV warpAffine image." << std::endl;
        return -1;
    }
}
