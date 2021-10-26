/**
 * @File    : compute_perspective_matrix.cpp
 * @Brief   : 已知原始坐标和变换后的坐标，通过方程方法计算投影矩阵
 *              利用 投影矩阵完成投影变换
 * @Author  : Wei Li
 * @Date    : 2021-09-14
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char **argv)
{
    // 1. 原始坐标和对应的变换后的坐标以 Point2f 进行存储
    cv::Point2f src[] = {cv::Point2f(0, 0), cv::Point2f(200.0, 0), cv::Point2f(0, 200.0), cv::Point2f(200, 200)};
    cv::Point2f dst[] = {cv::Point2f(100, 20), cv::Point2f(200, 20), cv::Point2f(50, 70), cv::Point2f(250, 70)};
    // compute perspective transform matrix
    cv::Mat perspective_transform_matrix = cv::getPerspectiveTransform(src, dst);
    std::cout << perspective_transform_matrix << std::endl;

    std::cout << "--------------------------" << std::endl;

    // 2. 原始坐标和对应的变换后的坐标以 Mat 进行存储
    cv::Mat src_idx = (cv::Mat_<float>(4, 2) << 0, 0, 200, 0, 0, 200, 200, 200);
    cv::Mat dst_idx = (cv::Mat_<float>(4, 2) << 100, 20, 200, 20, 50, 70, 250, 70);
    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_idx, dst_idx);
    std::cout << perspective_matrix << std::endl;

    // 3. 利用 投影矩阵完成投影变换
    if (argc > 1)
    {
        cv::Mat image = cv::imread(argv[1], 0);
        if (!image.data)
        {
            std::cout << "Error: reading image unsuccesfully." << std::endl;
            return -1;
        }
        cv::imshow("OriginImage", image);

        // 0. 计算投影矩阵
        cv::Point2f src[] = {cv::Point2f(0, 0), cv::Point2f(image.cols - 1, 0), cv::Point2f(0, image.rows - 1), cv::Point2f(image.cols - 1, image.rows - 1)};
        cv::Point2f dst[] = {cv::Point2f(50, 50), cv::Point2f(image.cols/3, 50), cv::Point2f(50, image.rows - 1), cv::Point2f(image.cols - 1, image.rows - 1)};
        // compute perspective transform matrix
        cv::Mat perspective_transform_matrix = cv::getPerspectiveTransform(src, dst);

        // 1. 利用 warpPerspective 完成投影变换
        cv::Mat dst_perspective_transform_img;
        cv::warpPerspective(image, 
                            dst_perspective_transform_img, 
                            perspective_transform_matrix, 
                            cv::Size(image.cols, image.rows),
                            1,
                            0,
                            cv::Scalar(125));
        cv::imshow("Image_perspective_transform", dst_perspective_transform_img);

        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    else
    {
        std::cout << "Usage: OpenCV warpAffine image." << std::endl;
        return -1;
    }

    return 0;
}
