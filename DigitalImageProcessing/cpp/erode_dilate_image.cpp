/**
 * @File    : erode_dilate_image.cpp
 * @Brief   : 形态学操作：腐蚀(选择一个任意领域[结构元]里面的最小值)和膨胀(选择一个任意领域[结构元]里面的最大值)
 * @Author  : Wei Li
 * @Date    : 2021-10-11
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// 2. 腐蚀操作
int radio = 1;     // 结构元半径
int Max_R = 20;   // 设置最大半径
cv::Mat d;       // 膨胀后图像
cv::Mat image;  // 输入图像

// 回调函数，调节 radio
void callBackRadio(int, void*)
{
    // 创建只有垂直方向的矩形结构元
    cv::Mat s = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 2*radio+1));
    // 进行膨胀操作
    cv::dilate(image, d, s);
    cv::imshow("DilateImage", d);
}

// -------------------------------
int main(int argc, char **argv)
{
    image = cv::imread(argv[1], 0);
    if (!image.data)
    {
        std::cout << "Error: Reading image file unsuccessfully.";
        return -1;
    }
    cv::imshow("OriginImage", image);

    // 1. 腐蚀操作
    // 创建结构元
    cv::Mat s = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat e;
    // 腐蚀操作，迭代次数(腐蚀次数) 2
    cv::erode(image, e, s, cv::Point(-1, -1), 2);
    cv::imshow("ErodeImage", e);

    // 2. 腐蚀操作
    // 创建显示膨胀效果的窗口
    cv::namedWindow("DilateWin", 1);
    // 创建调节 半径 radio 的进度条
    cv::createTrackbar("Radio", "DilateWin", &radio, Max_R, callBackRadio);
    callBackRadio(0, 0);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
