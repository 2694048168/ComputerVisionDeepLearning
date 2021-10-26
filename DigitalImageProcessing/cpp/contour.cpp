/**
 * @File    : contour.cpp
 * @Brief   : 轮廓：
        查找、绘制轮廓
        外包、拟合轮廓
        轮廓的周长和面积
        点和轮廓的位置关系
        轮廓的凸包缺陷
 * @Author  : Wei Li
 * @Date    : 2021-10-19
*/

#include <iostream>
#include <vector>
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

        cv::GaussianBlur(image, image, cv::Size(3, 3), 0.5);
        cv::Mat binaryImg;
        cv::Canny(image, binaryImg, 50, 200);
        cv::imshow("EdgeImage", binaryImg);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        int num = contours.size();
        for (size_t i = 0; i < num; ++i)
        {
            cv::Rect rect = cv::boundingRect(contours[i]);
            if (rect.area() > 1000)
            {
                cv::rectangle(image, rect, cv::Scalar(255));
            }
        }
        cv::imshow("Image", image);

        // // ------------ step 1, 查找、绘制轮廓 ------------
        // cv::findContours()
        // cv::drawContours()
        // // ------------ step 2, 外包、拟合轮廓 ------------
        // cv::approxPolyDP()
        // cv::boundingRect()
        // // ------------ step 3, 轮廓的周长和面积 ------------
        // cv::arcLength()
        // cv::contourArea()
        // // ------------ step 4, 点和轮廓的位置关系 ----------
        // cv::pointPolygonTest()
        // // ------------ step 5, 轮廓的凸包缺陷 ------------
        // cv::convexityDefects()

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
