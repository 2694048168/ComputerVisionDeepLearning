/**
 * @File    : conv_fft_theorem.cpp
 * @Brief   : 卷积定理: 卷积定义和傅里叶变换的关系(利用快速傅里叶变换)
 * @     对于卷积核为任意尺寸或者锚点在任意位置的情况，只是最后的裁剪部分不同。
 * @   虽然通过定义计算卷积比较耗时，但是当卷积核较小时，通过快速傅里叶变换计算卷积并没有明显的优势;
 * @   只有当卷积核较大时，利用傅里叶变换的快速算法计算卷积才会表现出明显的优势。
 * @Author  : Wei Li
 * @Date    : 2021-10-21
*/

#include <iostream>

cv::Mat FFT2Conv(cv::Mat image, cv::Mat kernel, int _borderType=cv::BORDER_DEFAULT, cv::Scalar _value=cv::Scalar())
{
    // step 0, 获取基本变量
    int R = image.rows;
    int C = image.cols;
    int r = kernel.rows;
    int c = kernel.cols;
    int tb = (r - 1) / 2;
    int lr = (c - 1) / 2;

    // step 1, 边界扩充
    cv::Mat img_padding;
    cv::copyMakeBorder(image, img_padding, tb, tb, lr, lr, _borderType, _value);

    // step 2, ZERO-padding
    int rows = cv::getOptimalDFTSize(img_padding.rows + r - 1);
    int cols = cv::getOptimalDFTSize(img_padding.cols + c - 1);
    cv::Mat img_zero_padding, kernel_zero;
    cv::copyMakeBorder(img_padding, img_zero_padding, 0, rows - img_padding.rows, 0, cols - img_padding.cols, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));
    cv::copyMakeBorder(kernel, kernel_zero, 0, rows - kernel.rows, 0, cols - kernel.cols,  cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));

    // step 3, 快速傅里叶变换
    cv::Mat fft_ipz, fft_kz;
    cv::dft(img_zero_padding, fft_ipz, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(kernel_zero, fft_kz, cv::DFT_COMPLEX_OUTPUT);

    // step 4, 两个傅里叶变换的点乘, 卷积定理
    cv::Mat ipz_kz;
    cv::mulSpectrum(fft_ipz, fft_kz, ipz_kz, cv::DFT_ROWS);

    // step 5, 傅里叶变换的逆变换
    cv::ifft_img;
    cv::dft(ipz_kz, ifft_img, cv::DFT_INVERSE + cv::DFT_SCALE + cv::DFT_REAL_OUTPUT);

    // step 6, 裁剪
    cv::Mat sameConv = ifft_img(cv::Rect(c - 1, r - 1, C + c -1, R + r - 1));
    return sameConv;
}

// --------------------------------
int main(int argc, char** argv)
{
    
    return 0;
}
