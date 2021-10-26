/**
 * @File    : compute_perspective_mouse.cpp
 * @Brief   : 利用 OpenCV 提供的鼠标事件，在原图和输出的画布上选择4组对应的坐标，
 *             计算投影变换矩阵并完成投影变换。
 * @Author  : Wei Li
 * @Date    : 2021-09-14
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat image;
// 投影变换后的图像
cv::Mat perspevtive_img;
// 原始图像与其对应坐标点
cv::Point2f image_point, perspevtive_img_point;
cv::Point2f src[4];
cv::Point2f dst[4];
int i = 0, j = 0;

// 通过鼠标事件，在原始图像中取四个坐标点
void mouse_image(int event, int x, int y, int flags, void *param)
{
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        // 记录坐标
        image_point = cv::Point2f(x, y);
        break;
    case cv::EVENT_LBUTTONUP:
        src[i] = image_point;
        cv::circle(image, src[i], 7, cv::Scalar(0), 3); // 标记
        ++i;
        break;
    default:
        break;
    }
}

// 通过鼠标事件，在输出图像中取四个坐标点
void mouse_perspective_img(int event, int x, int y, int flags, void *param)
{
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        // 记录坐标
        perspevtive_img_point = cv::Point2f(x, y);
        break;
    case cv::EVENT_LBUTTONUP:
        dst[j] = perspevtive_img_point;
        cv::circle(perspevtive_img, dst[j], 7, cv::Scalar(0), 3); // 标记
        ++j;
        break;
    default:
        break;
    }
}

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        image = cv::imread(argv[1], 0);
        if (!image.data)
        {
            std::cout << "Error: reading image unsuccesfully." << std::endl;
            return -1;
        }
        // 根据原始图像构建输出图像
        perspevtive_img = 255 * cv::Mat::ones(image.size(), CV_8UC1);

        // 原始图像的窗体上，定义鼠标事件
        const cv::String origin_winname = "Origin_image";
        cv::namedWindow(origin_winname, 1);
        cv::setMouseCallback(origin_winname, mouse_image, nullptr);

        // 输出图像的窗体上，定义鼠标事件
        const cv::String perspective_winname = "Perspective_img";
        cv::namedWindow(perspective_winname, 1);
        cv::setMouseCallback(perspective_winname, mouse_perspective_img, nullptr);

        cv::imshow(origin_winname, image);
        cv::imshow(perspective_winname, perspevtive_img);

        while (!(i == 4 && j == 4))
        {
            cv::imshow(origin_winname, image);
            cv::imshow(perspective_winname, perspevtive_img);
            if (cv::waitKey(50) == 'q')
            {
                break;
            }
        }
        cv::imshow(origin_winname, image);
        cv::imshow(perspective_winname, perspevtive_img);

        // 移除鼠标事件
        cv::setMouseCallback(origin_winname, nullptr, nullptr);
        cv::setMouseCallback(perspective_winname, nullptr, nullptr);

        // 计算投影变换矩阵
        cv::Mat perspective_transform_matrix = cv::getPerspectiveTransform(src, dst);
        // 利用 warpPerspective 完成投影变换
        cv::Mat dst_perspective_transform_img;
        cv::warpPerspective(image,
                            dst_perspective_transform_img,
                            perspective_transform_matrix,
                            perspevtive_img.size());
        cv::imshow("Image_perspective_transform", dst_perspective_transform_img);

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
