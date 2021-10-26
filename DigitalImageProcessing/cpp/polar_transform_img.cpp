/**
 * @File    : polar_transform_img.cpp
 * @Brief   : 利用极坐标变换对图像进行变换，校正图像中的圆形区域
 * OpenCV 实现了线性极坐标变换和对数极坐标变换。
 * @Author  : Wei Li
 * @Date    : 2021-09-15
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat polar(cv::Mat input_img, cv::Point2f center, cv::Size size, float minR=0, float minTheta=0, float thetaStep=1.0/4, float rStep=1.0)
{
    // 1. 构建 r
    cv::Mat ri = cv::Mat::zeros(cv::Size(1, size.height), CV_32FC1);
    for (int i = 0; i < size.height; ++i)
    {
        ri.at<float>(i, 0) = minR + i * rStep;
    }
    cv::Mat r = cv::repeat(ri, 1, size.width);

    // 2. 构建 theta
    cv::Mat thetaj = cv::Mat::zeros(cv::Size(size.width, 1), CV_32FC1);
    for (int j = 0; j < size.width; ++j)
    {
        thetaj.at<float>(0, j) = minTheta + j*thetaStep;
    }
    cv::Mat theta = cv::repeat(thetaj, size.height, 1);

    // 3. 将极坐标转换为笛卡尔坐标
    cv::Mat x, y;
    cv::polarToCart(r, theta, x, y, true);
    // 将坐标原点移动到中心点
    x += center.x;
    y += center.y;
    // 最邻近插值
    cv::Mat dst = 125 * cv::Mat::ones(size, CV_8UC1);
    for (int i = 0; i < size.height; ++i)
    {
        for (int j = 0; j < size.width; ++j)
        {
            float xij = x.at<float>(i, j);
            float yij = y.at<float>(i, j);
            int nearestx = int(std::round(xij));
            int nearesty = int(std::round(yij));
            if ((0 <= nearestx && nearestx < input_img.cols) && (0 <= nearesty && nearesty < input_img.rows))
            {
                dst.at<uchar>(i, j) = input_img.at<uchar>(nearestx, nearesty);
            }
        }
    }
    return dst;
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

        // 图像的极坐标变换
        float thetaStep = 1.0 / 4;
        float minR = 270;
        cv::Size size(int(360 / thetaStep), 70);
        cv::Mat dst = polar(image, cv::Point2f(508, 503), size, minR);
        
        // 沿着水平方向的镜像处理，翻转
        cv::flip(dst, dst, 0);
        cv::imshow("OriginImage", image);
        cv::imshow("PolarImage", dst);

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
