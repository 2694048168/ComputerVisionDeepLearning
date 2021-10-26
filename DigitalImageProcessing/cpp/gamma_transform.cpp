/**
 * @File    : gamma_transform.cpp
 * @Brief   : 图像的伽马变换，可以实现全局或者局部的对比度增强，亮度增大，人眼观察的视觉更多
 * @Author  : Wei Li
 * @Date    : 2021-09-16
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

        // 图像的伽马变换的实质就是对图像矩阵中的每一个数值进行幂运算
        // 1. 图像归一化操作
        cv::Mat normImage;
        image.convertTo(normImage, CV_64F, 1.0 / 255, 0);
        // 伽马变换
        double gamma = 0.5;
        cv::Mat output_gamma_img;
        cv::pow(normImage, gamma, output_gamma_img);
        cv::imshow("GammaTransform", output_gamma_img);
        // 如果没有数据类型转换，则直接保存浮点型的图像，这样虽然不会报错，但是保存后图像呈现黑色，看不到任何信息。
        // 伽马变换在提升对比度上有比较好的效果，但是需要手动调节 gamma 值。
        // 一种利用图像的直方图自动调节图像对比度的方法。
        output_gamma_img.convertTo(output_gamma_img, CV_8U, 255, 0);
        cv::imwrite("./output_gamma.png", output_gamma_img);

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
