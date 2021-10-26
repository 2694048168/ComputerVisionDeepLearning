/**
 * @File    : hist_equalization.cpp
 * @Brief   : 灰度图像的直方图均衡化(全局直方图均衡化)
 * @Author  : Wei Li
 * @Date    : 2021-09-16
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

cv::Mat equalHist(cv::Mat image)
{
    // 对于直方图均衡化的实现主要分四个步骤:
    CV_Assert(image.type() == CV_8UC1);
    // 1. 计算图像的灰度直方图。
    int rows = image.rows;
    int cols = image.cols;
    cv::Mat grayHist = calcGrayHist(image);
    // 2. 计算灰度直方图的累加直方图。
    cv::Mat zeroCumuMoment = cv::Mat::zeros(cv::Size(256, 1), CV_32SC1);
    for (size_t p = 0; p < 256; ++p)
    {
        if (p == 0)
        {
            zeroCumuMoment.at<int>(0, p) = grayHist.at<int>(0, 0);
        }
        else
        {
            zeroCumuMoment.at<int>(0, p) = zeroCumuMoment.at<int>(0, p - 1) + grayHist.at<int>(0, p);
        }
    }
    // 3. 根据累加直方图和直方图均衡化原理得到输入灰度级和输出灰度级之间的映射关系。
    cv::Mat output_q = cv::Mat::zeros(cv::Size(256, 1), CV_8UC1);
    float cofficient = 256.0 / (rows * cols);
    for (size_t p = 0; p < 256; ++p)
    {
        float q = cofficient * zeroCumuMoment.at<int>(0, p) - 1;
        if (q >= 0)
        {
            output_q.at<uchar>(0, p) = uchar(float(q));
        }
        else
        {
            output_q.at<uchar>(0, p) = 0;
        }
    }
    // 4. 根据第三步得到的灰度级映射关系，循环得到输出图像的每一个像素的灰度级。
    cv::Mat equalHistImage = cv::Mat::zeros(image.size(), CV_8UC1);
    for (size_t idx_row = 0; idx_row < rows; ++idx_row)
    {
        for (size_t idx_col = 0; idx_col < cols; ++idx_col)
        {
            int p = image.at<uchar>(idx_row, idx_col);
            equalHistImage.at<uchar>(idx_row, idx_col) = output_q.at<uchar>(0, p);
        }
    }

    return equalHistImage;
}

// -------------------------------
int main(int argc, char **argv)
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

        // 1. 图像直方图均衡化
        cv::Mat equalizationHistImg = equalHist(image);
        cv::imshow("EqualizeHistImg", equalizationHistImg);

        // 2. 自适应直方图均衡化 ----> 限制对比度的自适应直方图均衡化
        // 构建 CLAHE 对象
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        cv::Mat dst_contrast_limit;
        // 限制对比度的自适应直方图均衡化
        clahe->apply(image, dst_contrast_limit);
        cv::imshow("ContrastLimitImage", dst_contrast_limit);

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
