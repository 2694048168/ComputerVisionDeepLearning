/**
 * @File    : convert_color.cpp
 * @Brief   : 色彩空间的转换; 调整彩色图像的饱和度和亮度
 * @Author  : Wei Li
 * @Date    : 2021-10-21
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// CONST
cv::Mat image;
cv::Mat image_norm;
cv::Mat image_hls;
cv::Mat image_ls;

int width, height;
cv::String win_name = "Saturation_Liminance_Image";
int L = 0;
int S = 0;
int MAX_VALUE = 100;
void callBack_LS(int, void*);

// -------------------------------
int main(int argc, char** argv)
{
    if (argc > 1)
    {
        image = cv::imread(argv[1]);
        if (!image.data)
        {
            std::cout << "Error: reading image unsuccesfully." << std::endl;
            return -1;
        }
        cv::imshow("OriginImage", image);

        width = image.cols;
        height = image.rows;
        image.convertTo(image_norm, CV_32FC3, 1.0 / 255, 0);
        cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);

        cv::createTrackbar("Saturation", win_name, &S, MAX_VALUE, callBack_LS);
        cv::createTrackbar("Liminance", win_name, &L, MAX_VALUE, callBack_LS);
        callBack_LS(0, 0);

        cv::waitKey(0);
        cv::destroyAllWindows();
        return 0;
    }
    else
    {
        std::cout << "Usage: OpenCV python script imageFile." << std::endl;
        return -1;
    }
}


void callBack_LS(int, void*)
{
    cv::cvtColor(image_norm, image_hls, cv::COLOR_BGR2HLS);
    for (size_t r = 0; r < height; ++r)
    {
        for (size_t c = 0; c < width; ++c)
        {
            cv::Vec3f hls = image_hls.at<cv::Vec3f>(r, c);
            hls = cv::Vec3f(hls[0], 
                            (1 + L / double(MAX_VALUE))*hls[1] > 1 ? 1 : (1 + L / double(MAX_VALUE)) * hls[1],
                            (1 + S / double(MAX_VALUE))*hls[2] > 1 ? 1 : (1 + S / double(MAX_VALUE)) * hls[2]);
                            
        image_hls.at<cv::Vec3f>(r, c) = hls;
        }
    }

    cv::cvtColor(image_hls, image_ls, cv::COLOR_HLS2BGR);
    cv::imshow(win_name, image_ls);
}
