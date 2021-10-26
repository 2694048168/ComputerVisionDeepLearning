/**
 * @File    : joint_bilateral_filter.cpp
 * @Brief   : 联合双边滤波
 * @Author  : Wei Li
 * @Date    : 2021-09-25
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat getClosenessWeight(double sigma_g, cv::Size size)
{
    int W = size.width;
    int H = size.height;
    int center_W = (W - 1) / 2;
    int center_H = (H - 1) / 2;
    // 空间权重模板
    cv::Mat closenessWeight = cv::Mat::zeros(size, CV_64FC1);
    for (size_t r = 0; r < H; ++r)
    {
        for (size_t c = 0; c < W; ++c)
        {
            double norm2 = cv::pow(double(r - center_H), 2.0) + cv::pow(double(c - center_W), 2.0);
            double sigma_g2 = 2.0 * std::pow(sigma_g, 2.0);
            closenessWeight.at<double>(r, c) = cv::exp(-norm2 / sigma_g2);
        }
    }
    return closenessWeight;
}

/*联合双边滤波
    Args:
        image (ndarray): 输入单通道图像, 灰度级范围[0，1]
        H ([int]): 权重模板的高
        W ([int]): 权重模板的宽
        sigma_g ([float]): 空间距离权重模板的标准差，大于 1
        sigma_d ([float]): 相似性权重模板的标准差， 小于 1

    Returns:
        [ndarray]: 双边滤波结果图像, 浮点型矩阵
*/
cv::Mat jointBilateralFilter(cv::Mat &image, cv::Size winSize, float sigma_g, float sigma_d, int borterType=cv::BORDER_DEFAULT)
{
    int winH = winSize.height;
    int winW = winSize.width;
    CV_Assert(winH > 0 && winW > 0);
    CV_Assert(winH % 2 == 1 && winW % 2 == 1);
    if (winH == 1 && winW == 1)
    {
        return image;
    }

    int half_winW = (winW - 1) / 2;
    int half_winH = (winH - 1) / 2;
    // 空间距离权重因子
    cv::Mat closenessWight = getClosenessWeight(sigma_g, winSize);
    // 对原始图像进行高斯平滑
    cv::Mat image_gauss;
    cv::GaussianBlur(image, image_gauss, winSize, sigma_g);
    // 对原始图像和高斯平滑后的图像进行边界扩充
    cv::Mat image_padding, image_gauss_padding;
    cv::copyMakeBorder(image, image_padding, half_winH, half_winH, half_winW, half_winW, borterType);
    cv::copyMakeBorder(image_gauss, image_gauss_padding, half_winH, half_winH, half_winW, half_winW, borterType);

    int rows = image.rows;
    int cols = image.cols;
    // 双边滤波结果
    cv::Mat joint_bilateral_filter_image = cv::Mat::ones(image.size(), CV_64FC1);
    int i = 0, j = 0;
    // 对每一个像素的邻域进行和卷积
    for (size_t r = half_winH; r < half_winH+rows; ++r)
    {
        for (size_t c = half_winW; c < half_winW+cols; ++c)
        {
            double pixel = image_gauss_padding.at<double>(r, c);
            // 截取当前位置的邻域
            cv::Mat region = image_gauss_padding(cv::Rect(c-half_winW, r-half_winH, winW, winH));
            // 当前位置的相似性权重
            cv::Mat similarityWeight;
            cv::pow(region - pixel, 2.0, similarityWeight);
            cv::exp(-0.5 * similarityWeight / cv::pow(sigma_d, 2.0), similarityWeight);
            // 两个权重模板点乘，然后归一化
            cv::Mat weight = closenessWight.mul(similarityWeight);
            weight = weight / cv::sum(weight)[0];
            // 权重模板和当前的邻域对应位置相乘，然后求和
            cv::Mat result = image_padding(cv::Rect(c - half_winW, r - half_winH, winW, winH));
            joint_bilateral_filter_image.at<double>(i, j) = cv::sum(result.mul(weight))[0];
            ++j;
        }
        j = 0;
        ++i;
    }
    return joint_bilateral_filter_image;
}

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

        // 数据类型转换
        cv::Mat image_datatype;
        image.convertTo(image_datatype, CV_64F, 1.0, 0);
        // 联合双边滤波
        cv::Mat joint_bilateral_img = jointBilateralFilter(image_datatype, cv::Size(33, 33), 7, 2);
        // 保存或者显示 8-bit 图像，灰度级[0-255]
        joint_bilateral_img.convertTo(joint_bilateral_img, CV_8U, 255, 0);
        cv::imshow("bilateral_blur_img", joint_bilateral_img);

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
