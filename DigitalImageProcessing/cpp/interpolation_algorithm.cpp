/**
 * @File    : interpolation_algorithm.cpp
 * @Brief   : 已知仿射变换矩阵，利用插值方法完成图像的几何变换(空间域操作)
 * @Author  : Wei Li
 * @Date    : 2021-09-14
*/


#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

        // 1. 利用 warpAffine 进行缩放
        cv::Mat affine_matrix_scale = (cv::Mat_<float>(2, 3) << 0.5, 0, 0, 0, 0.5, 0);
        cv::Mat dst_affine_img_scale;
        // 利用 OpenCV 提供的 API，函数原型和参数含义查看 API 接口即可
        cv::warpAffine(image, dst_affine_img_scale, affine_matrix_scale, cv::Size(image.cols / 2, image.rows / 2));
        cv::imshow("Image_affine_scale_2", dst_affine_img_scale);

        // 2. 利用 resize 进行缩放，不需要创建仿射变换矩阵
        cv::Mat dst_resize_img_scale;
        cv::resize(image, dst_resize_img_scale, cv::Size(image.cols / 2, image.rows / 2), 0.5, 0.5);
        cv::imshow("Image_resize_scale_2", dst_resize_img_scale);

        // 3. 利用 rotate 进行旋转，不是通过仿射变换矩阵实现的，而是通过类似矩阵转置方式的进行行列交换实现的
        cv::Mat dst_rotate_img;
        // 顺时针旋转 90 度
        cv::rotate(image, dst_rotate_img, cv::ROTATE_90_CLOCKWISE);
        cv::imshow("Image_rotate", dst_rotate_img);

        cv::Mat dst_rotate_scale_img;
        // 顺时针旋转 90 度
        cv::rotate(dst_resize_img_scale, dst_rotate_scale_img, cv::ROTATE_90_CLOCKWISE);
        cv::imshow("Image_rotate", dst_rotate_scale_img);

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
