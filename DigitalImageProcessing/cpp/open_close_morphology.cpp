/**
 * @File    : open_close_morphology.cpp
 * @Brief   : 形态学操作：开运算(腐蚀后膨胀)和闭运算(膨胀后腐蚀); 顶帽变换、底帽变换、形态学梯度
 * @Author  : Wei Li
 * @Date    : 2021-10-13
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// 开运算和闭运算
int r = 1;                       // 结构元半径
int i = 1;                      // 迭代次数
int Max_R = 20;                // 设置最大半径
int Max_I = 20;               // 设置最大迭代次数
cv::Mat morphologyImg;       // 膨胀后图像
cv::Mat image;              // 输入图像

// 回调函数，调节半径 r 和迭代次数 i
void callBack(int, void*)
{
    // 创建矩形结构元
    cv::Mat s = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*r+1, 2*r+1));
    // 进行开运算和闭运算
    // cv.morphologyEx (src, dst, op, kernel, anchor = new cv.Point(-1, -1), iterations = 1, 
                        // borderType = cv.BORDER_CONSTANT, borderValue = cv.morphologyDefaultBorderValue())
    // 1. 开运算
    // cv::morphologyEx(image, morphologyImg, cv::MORPH_OPEN, s, cv::Point(-1, -1), i);

    // 2. 闭运算
    // cv::morphologyEx(image, morphologyImg, cv::MORPH_CLOSE, s, cv::Point(-1, -1), i);

    // 3. 顶帽变换
    // cv::morphologyEx(image, morphologyImg, cv::MORPH_TOPHAT, s, cv::Point(-1, -1), i);

    // 4. 底帽变换
    // cv::morphologyEx(image, morphologyImg, cv::MORPH_BLACKHAT, s, cv::Point(-1, -1), i);

    // 5. 形态学梯度
    cv::morphologyEx(image, morphologyImg, cv::MORPH_GRADIENT, s, cv::Point(-1, -1), i);

    cv::imshow("morphology", morphologyImg);
}

// -------------------------------
int main(int argc, char **argv)
{
    image = cv::imread(argv[1], 0);
    if (!image.data)
    {
        std::cout << "Error: Reading image file unsuccessfully.";
        return -1;
    }
    cv::imshow("OriginImage", image);

    // 创建显示膨胀效果的窗口
    cv::namedWindow("morphology", 1);
    // 创建调节 半径 radio 的进度条
    cv::createTrackbar("r", "morphology", &r, Max_R, callBack);
    cv::createTrackbar("i", "morphology", &i, Max_I, callBack);
    callBack(0, 0);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
