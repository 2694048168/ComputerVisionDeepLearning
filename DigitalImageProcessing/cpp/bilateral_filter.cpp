/**
 * @File    : bilateral_filter.cpp
 * @Brief   : 双边滤波; 非常耗时，提出双边滤波的快速算法
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

/*双边滤波
    Args:
        image (ndarray): 输入单通道图像, 灰度级范围[0，1]
        H ([int]): 权重模板的高
        W ([int]): 权重模板的宽
        sigma_g ([float]): 空间距离权重模板的标准差，大于 1
        sigma_d ([float]): 相似性权重模板的标准差， 小于 1

    Returns:
        [ndarray]: 双边滤波结果图像, 浮点型矩阵
*/
cv::Mat bilateralFilterGray(const cv::Mat &image, cv::Size winSize, float sigma_g, float sigma_d)
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
    int rows = image.rows;
    int cols = image.cols;
    // 双边滤波结果
    cv::Mat bilateral_filter_image = cv::Mat::ones(image.size(), CV_32FC1);
    // 对每一个像素的邻域进行和卷积
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            double pixel = image.at<double>(r, c);
            // 判断边界
            int rTop = (r - half_winH) < 0 ? 0 : r - half_winH;
            int rBottom = (r + half_winH) < rows - 1 ? rows - 1 : r + half_winH;
            int cLeft = (c - half_winW) < 0 ? 0 : c - half_winW;
            int cRight = (c + half_winW) < cols - 1 ? cols - 1 : c + half_winW;
            // 核作用区域
            cv::Mat region = image(cv::Rect(cv::Point(cLeft, rTop), cv::Point(cRight + 1, rBottom + 1))).clone();
            // 相似性权重模板
            cv::Mat similarityWeight;
            cv::pow(region - pixel, 2.0, similarityWeight);
            cv::exp(-0.5 * similarityWeight / std::pow(sigma_d, 2), similarityWeight);
            similarityWeight /= cv::pow(sigma_d, 2);
            // 空间距离权重
            cv::Rect regionRect = cv::Rect(cv::Point(cLeft - c + half_winW, rTop - r + half_winH), cv::Point(cRight - c + half_winW + 1, rBottom - r + half_winH + 1));
            cv::Mat closenessWightTemp = closenessWight(regionRect).clone();
            // 两个权重模板点乘，然后归一化
            cv::Mat weightTemp = closenessWightTemp.mul(similarityWeight);
            weightTemp = weightTemp / cv::sum(weightTemp)[0];
            // 权重模板和当前的邻域对应位置相乘，然后求和
            cv::Mat result = weightTemp.mul(region);
            bilateral_filter_image.at<float>(r, c) = cv::sum(result)[0];
        }
    }
    return bilateral_filter_image;
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

        // 灰度值归一化
        cv::Mat image_norm;
        image.convertTo(image_norm, CV_64FC1, 1.0/255, 0);
        // 双边滤波
        cv::Mat bilateral_blur_img = bilateralFilterGray(image_norm, cv::Size(33, 33), 19, 0.5);
        // 保存或者显示 8-bit 图像，灰度级[0-255]
        bilateral_blur_img.convertTo(bilateral_blur_img, CV_8U, 255, 0);
        cv::imshow("bilateral_blur_img", bilateral_blur_img);

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
