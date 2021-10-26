/**
 * @File    : global_threshold.cpp
 * @Brief   : 全局阈值分割
 * @Author  : Wei Li
 * @Date    : 2021-09-28
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

        // 1. 手动设置阈值
        double thresholdVal = 150;
        cv::Mat dst_img;
        cv::threshold(image, dst_img, thresholdVal, 255, cv::THRESH_BINARY);
        cv::imshow("binary_threshold", dst_img);

        // 2. Otsu 算法确定阈值
        double otsuThreshold = 0;
        cv::Mat dst_Otsu;
        double otsuThresholdVal = cv::threshold(image, dst_Otsu, otsuThreshold, 255, cv::THRESH_OTSU + cv::THRESH_BINARY);
        std::cout << "The Otsu threshold value is : " << otsuThresholdVal << std::endl;
        cv::imshow("Otsu_threshold", dst_Otsu);

        // 3. TRIANGLE 算法
        double triThreshold = 0;
        cv::Mat dst_tri;
        double triThresholdVal = cv::threshold(image, dst_tri, triThreshold, 255, cv::THRESH_TRIANGLE + cv::THRESH_BINARY);
        std::cout << "The TRIANGLE threshold value is : " << triThresholdVal << std::endl;
        cv::imshow("tri_threshold", dst_tri);

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
