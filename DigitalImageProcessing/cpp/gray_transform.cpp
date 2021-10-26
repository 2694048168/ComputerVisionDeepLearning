/**
 * @File    : gray_transform.cpp
 * @Brief   : 线性变换进行对比度增强；直方图正规化
 * @Author  : Wei Li
 * @Date    : 2021-09-16
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

        // 1. OpenCV 实现线性变换增对比度 cv::Mat::convertTo()
        cv::Mat linear_transform_img;
        image.convertTo(linear_transform_img, CV_8UC1, 1.5, 0);
        cv::imshow("LinearTranformImg", linear_transform_img);

        // 2. OpenCV 实现线性变换增对比度 注意数据格式以及计算后进行的饱和操作
        cv::Mat linear_transform = image * 1.5;
        cv::imshow("LinearTransform", linear_transform);

        // 3. OpenCV 实现线性变换增对比度 cv::convertScaleAbs()
        cv::Mat linear_transform_func;
        cv::convertScaleAbs(image, linear_transform_func, 1.5, 0);
        cv::imshow("LinearTransformFunc", linear_transform_func);

        // 4. 直方图正规化
        // 计算原始图像中的最大值和最小值
        double pixelMin, pixelMax;
        cv::minMaxLoc(image, &pixelMin, &pixelMax, nullptr, nullptr);
        // 设置输出图像的最大和最小灰度级
        double outputImageMin = 0, outputImageMax = 255;
        // 计算线性变换的系数
        double a = (outputImageMax - outputImageMin) / (pixelMax - pixelMin);
        double b = outputImageMin - a * pixelMin;
        // 进行图像的线性变换
        cv::Mat normLinearImage;
        cv::convertScaleAbs(image, normLinearImage, a, b);
        cv::imshow("NormLinearImage", normLinearImage);

        // 5. 利用 OpenCV 正规化函数实现直方图的正规化
        // 利用函数可以直接处理多通道矩阵(图像)
        cv::Mat dstNormImg;
        cv::normalize(image, dstNormImg, 255, 0, cv::NORM_MINMAX, CV_8U);
        cv::imshow("NormalizeImage", dstNormImg);

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
