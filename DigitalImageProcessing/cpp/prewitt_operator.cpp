/**
 * @File    : prewitt_operator.cpp
 * @Brief   : Prewitt Operator(Algorithm) —— 可分离卷积核
 *              Prewitt算子均是可分离的，为了减少耗时，
 *          在代码实现中, 利用卷积运算的结合律先进行水平方向上的平滑，再进行垂直方向上的差分，
 *          或者先进行垂直方向上的平滑，再进行水平方向上的差分
 * @Author  : Wei Li
 * @Date    : 2021-10-13
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void PrewittOperator(cv::InputArray src, cv::OutputArray dst, int ddepth, int x, int y=0, int borderType=cv::BORDER_DEFAULT)
{
    CV_Assert(!(x==0 && y==0));

    // x 不等于 0, src 和 prewitt_x 卷积核进行卷积
    if(x != 0 && y == 0)
    {
        // 可分离 prewitt_x 卷积核
        cv::Mat prewitt_x_y = (cv::Mat_<float>(3, 1) << 1, 1, 1);
        cv::Mat prewitt_x_x = (cv::Mat_<float>(1, 3) << 1, 0, -1);
        // 可分离的离散二维卷积
        cv::Mat prewitt_x_img;
        cv::Mat prewitt_x_flip, prewitt_y_flip;
        cv::flip(prewitt_x_y, prewitt_x_flip, -1);
        cv::flip(prewitt_x_x, prewitt_y_flip, -1);
        cv::filter2D(src, prewitt_x_img, ddepth, prewitt_x_flip, cv::Point(-1, -1), borderType);
        cv::filter2D(prewitt_x_img, dst, ddepth, prewitt_y_flip, cv::Point(-1, -1), borderType);
    }

    // y 不等于 0, src 和 prewitt_y 卷积核进行卷积
    if(y != 0 && x == 0)
    {
        // 可分离 prewitt_x 卷积核
        cv::Mat prewitt_y_x = (cv::Mat_<float>(3, 1) << 1, 1, 1);
        cv::Mat prewitt_y_y = (cv::Mat_<float>(1, 3) << 1, 0, -1);
        // 可分离的离散二维卷积
        cv::Mat prewitt_y_img;
        cv::Mat prewitt_y_flip, prewitt_x_flip;
        cv::flip(prewitt_y_x, prewitt_x_flip, -1);
        cv::flip(prewitt_y_y, prewitt_y_flip, -1);
        cv::filter2D(src, prewitt_y_img, ddepth, prewitt_x_flip, cv::Point(-1, -1), borderType);
        cv::filter2D(prewitt_y_img, dst, ddepth, prewitt_y_flip, cv::Point(-1, -1), borderType);
    }
}

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

    // 原始图像和 prewitt kernerl-x 卷积
    cv::Mat img_prewitt_x;
    PrewittOperator(image, img_prewitt_x, CV_32FC1, 1, 0);

    // 原始图像和 prewitt kernerl-y 卷积
    cv::Mat img_prewitt_y;
    PrewittOperator(image, img_prewitt_y, CV_32FC1, 0, 1);

    // 两个卷积结果的灰度级显示
    cv::Mat abs_img_prewitt_x, abs_img_prewitt_y;
    cv::convertScaleAbs(img_prewitt_x, abs_img_prewitt_x, 1, 0);
    cv::convertScaleAbs(img_prewitt_y, abs_img_prewitt_y, 1, 0);
    cv::imshow("horizontal_edge", abs_img_prewitt_x);
    cv::imshow("vertical_edge", abs_img_prewitt_y);

    // 根据 prewiit 两个卷积结果，获取最终图像的边缘强度
    cv::Mat edge;
    cv::Mat img_roberts_x_x, img_roberts_y_y;
    cv::pow(img_prewitt_x, 2.0, img_roberts_x_x);
    cv::pow(img_prewitt_y, 2.0, img_roberts_y_y);
    cv::sqrt(img_roberts_x_x + img_roberts_y_y, edge);
    // 数据类型转换，边缘强度的灰度级显示
    edge.convertTo(edge, CV_8UC1);
    cv::imshow("edge", edge);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
