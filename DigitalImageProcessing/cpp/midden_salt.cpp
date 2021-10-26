/**
 * @File    : midden_salt.cpp
 * @Brief   : 椒盐噪声; 中值滤波(平滑)-非线性滤波器
 * @Author  : Wei Li
 * @Date    : 2021-09-24
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat medianSmooth(cv::Mat img, cv::Size winSize, int borderType=cv::BORDER_DEFAULT)
{
    CV_Assert(img.type() == CV_8UC1);
    int H = winSize.height;
    int W = winSize.width;
    CV_Assert(H > 0 && W > 0);
    CV_Assert(H % 2 == 1 && W % 2 == 1);
    int h = (H - 1) / 2;
    int w = (W - 1) / 2;

    cv::Mat paddImg;
    cv::copyMakeBorder(img, paddImg, h, h, w, w, borderType);

    int rows = img.rows;
    int cols = img.cols;
    cv::Mat medianImg(img.size(), CV_8UC1);

    int i = 0, j = 0;
    // 中数的位置
    int index = (H*W - 1) / 2;
    for (size_t r = 0; r < h + rows; ++r)
    {
        for (size_t c = 0; c < w + cols; ++c)
        {
            // 取以当前位置为中心，大小为 winSize 的邻域
            cv::Mat region = paddImg(cv::Rect(c-w, r-h, W, H)).clone();
            // 将该邻域转换为行矩阵
            region = region.reshape(1, 1);
            // 排序
            cv::sort(region, region, cv::SORT_EVERY_ROW);
            // 取中数
            uchar medianValue = region.at<uchar>(0, index);
            medianImg.at<uchar>(i, j) = medianValue;
            ++j;
        }
        ++i;
        j = 0;
    }
    return medianImg;
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

        // 1. OpenCV 函数进行中值滤波器
        cv::Mat median_blur_img = medianSmooth(image, cv::Size(5, 5));
        cv::imshow("median_blur_img", median_blur_img);

        // 2. 利用 OpenCV 提供的函数，计算效率
        cv::Mat median_dst;
        cv::medianBlur(image, median_dst, 5);
        cv::imshow("medianImg", median_dst);

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
