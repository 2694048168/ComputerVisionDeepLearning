/**
 * @File    : laplace_operator.cpp
 * @Brief   : 拉普拉斯二维微分算子 —— 不可分离的单独一个卷积
             拉普拉斯算子对噪声很敏感，使用首先应对图像进行高斯平滑，然后再与拉普拉斯算子卷积，最后得到二值化边缘图。
            水墨效果的边缘图，该边缘图也在某种程度上体现了边缘强度。
 * @Author  : Wei Li
 * @Date    : 2021-10-17
*/

#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


// -------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread(argv[1], 0);
    if (!image.data)
    {
        std::cout << "Error: Reading image file unsuccessfully.";
        return -1;
    }
    cv::imshow("OriginImage", image);

    // -------- Laplace Operator --------
    // step 1, 高斯平滑
    // cv::GaussianBlur(image, image, cv::Size(7, 1), 1, 0);

    // step 2, 拉普拉斯卷积
    cv::Mat imgLaplaceConv;
    // void Laplacian(src, dst, ddepth, ksize=1, scale=1, delta=0, borderType=cv::BORDER_DEFAULT);
    cv::Laplacian(image, imgLaplaceConv, CV_32F, 3);

    // case 1, 阈值化处理
    cv::convertScaleAbs(imgLaplaceConv, imgLaplaceConv, 1.0, 0);
    // 做反色处理，以黑色显示边缘
    imgLaplaceConv = 255 - imgLaplaceConv;
    cv::imshow("LaplaceEdge", imgLaplaceConv);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
