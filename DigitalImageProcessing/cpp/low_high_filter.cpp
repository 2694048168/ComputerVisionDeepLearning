/**
 * @File    : low_high_filter.cpp
 * @Brief   : 低通滤波器和高通滤波器(理想滤波器；巴特沃斯滤波器；高斯滤波器)
 * @Author  : Wei Li
 * @Date    : 2021-10-22
*/


#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


/**---------------------------------------------------------------------------------------------------
 * 对于低通滤波的实现，将使用求傅里叶谱的函数 amplitudeSpectrum 和傅里叶谱的灰度级显示函数 graySpectrum。
 * 注意，傅里叶谱在低通滤波中并没有起到任何作用，
 * 只是通过傅里叶谱的灰度级显示来观察低通滤波器与傅里叶变换点乘后的灰度级显示是怎样的效果。
 * 为了同时观察三种滤波器和截断频率对低通滤波效果的影响，在实现中加人了两个进度条来实时调整这两个参数。
 * ---------------------------------------------------------------------------------------------------
 */
cv::Mat image;                                          // 输入图像矩阵
cv::Mat image_fourier;                                  // 图像的快速傅里叶变换
cv::Point maxLoc;                                       // 傅里叶谱的最大值的坐标
int radius = 20;                                        // 滤波器截至频率
const int Max_RADIUS = 100;                             // 设置最大截至频率
cv::Mat lpFilter;                                       // 低通滤波器
int lpType = 0;                                         // 低通滤波器类型
const int MAX_LPTYPE = 2;                               // 低通滤波器类型的总数量
cv::Mat F_lpFilter;                                     // 低通傅里叶变换
cv::Mat FlpSpectrum;                                    // 低通傅里叶变换的傅里叶谱的灰度级
cv::Mat result;                                         // 低通滤波器的效果
cv::String lpFilterspectrum = "LowPassFourierSpectrum"; // 显示窗口的名称

// 低通滤波器类型
enum LP_FILTER_TYPE
{
    ILP_FILTER = 0,
    BLP_FILTER = 1,
    GLP_FILTER = 2
};

// 快速傅里叶变换
void fft2Image(cv::InputArray _src, cv::OutputArray _dst);
// 幅度谱
void amplitudeSpectrum(cv::InputArray _srcFFT, cv::OutputArray _dstSpectrum);
// 幅度谱的灰度级显示
cv::Mat graySpectrum(cv::Mat spectrum);

// 回调函数
void callback_lpFilter(int, void *);
// 构建低通滤波器
cv::Mat createLPFilter(cv::Size size, cv::Point center, float radius, int type, int n = 2);

// -----------------------------------------------------------------------------------------
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
        cv::imshow("OriginImage", image);

        // ------------ step 1, reading image file to float data type ------------
        cv::Mat image_float;
        image.convertTo(image_float, CV_32FC1, 1.0, 0.0);

        // ------------ step 2, each pixel value multiply (-1)^(r+c) ------------
        for (size_t r = 0; r < image_float.rows; ++r)
        {
            for (size_t c = 0; c < image_float.cols; ++c)
            {
                if ((r + c) % 2)
                {
                    image_float.at<float>(r, c) *= -1;
                }
            }
        }

        // ------------ step 3 and step 4, zero-padding and FFT ------------
        fft2Image(image_float, image_fourier);
        cv::Mat amplSpec;
        amplitudeSpectrum(image_fourier, amplSpec);
        cv::Mat spectrum = graySpectrum(amplSpec);
        cv::imshow("OriginalFFTSpectrum", spectrum);
        // cv::imwrite("./image/original_spectrum.png", spectrum);

        cv::minMaxLoc(spectrum, nullptr, nullptr, nullptr, &maxLoc);

        // ------------ Low Pass Filter ------------
        cv::namedWindow(lpFilterspectrum, cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("lpType", lpFilterspectrum, &lpType, MAX_LPTYPE, callback_lpFilter);
        cv::createTrackbar("radius", lpFilterspectrum, &radius, Max_RADIUS, callback_lpFilter);
        callback_lpFilter(0, 0);

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

/**低通滤波器 low pass filter
 * 参数 size 代表滤波器的尺寸，即快速傅里叶变换的尺寸; 
 * 参数 center 代表傅里叶谱的中心位置（即:最大值的位置); 
 * 参数 radius 代表截断频率;
 * 参数 type 代表所定义的枚举类型，enum LPFILTER_TYPE {ILP_FILTER=0, BLP_FILTER=1, GLP_FILTER=2}代表三种不同的低通滤波器; 
 * 参数 n 只有在构建巴特沃斯滤波器时才用到的参数，表示滤波器的阶数。
 */
cv::Mat createLPFilter(cv::Size size, cv::Point center, float radius, int type, int n)
{
    cv::Mat lpFilter = cv::Mat::zeros(size, CV_32FC1);
    int rows = size.height;
    int cols = size.width;
    if (radius <= 0)
    {
        return lpFilter;
    }

    // case 1, 理想低通滤波器
    if (type == ILP_FILTER)
    {
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                float norm2 = std::pow(std::abs(float(r - center.y)), 2) + std::pow(std::abs(float(c - center.x)), 2);
                if (std::sqrt(norm2) < radius)
                {
                    lpFilter.at<float>(r, c) = 1;
                }
                else
                {
                    lpFilter.at<float>(r, c) = 0;
                }
            }
        }
    }

    // case 2, 巴特沃斯低通滤波器
    if (type == BLP_FILTER)
    {
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                lpFilter.at<float>(r, c) = float(1.0 / (1.0 + std::pow(std::sqrt(std::pow(r - center.y, 2.0) + std::pow(c - center.x, 2.0)) / radius, 2.0 * n)));
            }
        }
    }

    // case 3, 高斯低通滤波器
    if (type == GLP_FILTER)
    {
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                lpFilter.at<float>(r, c) = float(std::exp(-(std::pow(c - center.x, 2.0) + std::pow(r - center.y, 2.0)) / (2 * std::pow(radius, 2.0))));
            }
        }
    }

    return lpFilter;
}

// 回调函数
void callback_lpFilter(int, void *)
{
    // ------------ step 5, 构建低通滤波器 ------------
    lpFilter = createLPFilter(image_fourier.size(), maxLoc, radius, lpType, 2);

    // ------------ step 6, 低通滤波器和图像的快速傅里叶变换进行点乘 ------------
    F_lpFilter.create(image_fourier.size(), image_fourier.type());
    for (size_t r = 0; r < F_lpFilter.rows; ++r)
    {
        for (size_t c = 0; c < F_lpFilter.cols; ++c)
        {
            // 分别取出当前位置的快速傅里叶变换和低通滤波器的值
            cv::Vec2f F_rc = image_fourier.at<cv::Vec2f>(r, c);
            float lpFilter_rc = lpFilter.at<float>(r, c);
            F_lpFilter.at<cv::Vec2f>(r, c) = F_rc * lpFilter_rc;
        }
    }

    amplitudeSpectrum(F_lpFilter, FlpSpectrum);
    FlpSpectrum = graySpectrum(FlpSpectrum);
    cv::imshow("lpFilterspectrum", FlpSpectrum);
    // cv::imwrite("./image/lpFilterspectrum.png", FlpSpectrum);

    // ------------ step 7 and stpe 8, 对低通滤波器傅里叶变换进行逆变换，取实部 ------------
    cv::dft(F_lpFilter, result, cv::DFT_SCALE + cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT);

    // ------------ step 9, multiply (-1)^(x+y) ------------
    for (size_t r = 0; r < result.rows; ++r)
    {
        for (size_t c = 0; c < result.cols; ++c)
        {
            if ((r + c) % 2)
            {
                result.at<float>(r, c) *= -1;
            }
        }
    }

    // NOTE: convert result to CV_8U(data type)
    result.convertTo(result, CV_8UC1, 1.0, 0);

    // ------------ step 10, 截取左上部分，其大小和输入原始图像一致 ------------
    result = result(cv::Rect(0, 0, image.cols, image.rows)).clone();
    cv::imshow("LPFilterImgae", result);
    // cv::imwrite("./image/LPFilterImgae.png", result);
}

// 高通滤波器 high pass filter

// ------------ 快速离散傅里叶变换 ------------
void fft2Image(cv::InputArray InputImage, cv::OutputArray img_fourier)
{
    cv::Mat image = InputImage.getMat();
    int rows = image.rows;
    int cols = image.cols;

    // 满足快速傅里叶变换的最优 行数 和 列数
    int row_padding = cv::getOptimalDFTSize(rows);
    int col_padding = cv::getOptimalDFTSize(cols);
    // 左侧和下侧 zero-padding
    cv::Mat fourier;
    cv::copyMakeBorder(image, fourier, 0, row_padding - rows, 0, col_padding - cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // 快速傅里叶变换
    cv::dft(fourier, img_fourier, cv::DFT_COMPLEX_OUTPUT);
}

// ------------ 傅里叶变换的两个度量: 幅度谱和相位谱 ------------
void amplitudeSpectrum(cv::InputArray _srcFFT, cv::OutputArray _dstSpectrum)
{
    // 实部和虚部两个通道
    CV_Assert(_srcFFT.channels() == 2);
    // 分离实部和虚部两个通道
    std::vector<cv::Mat> FFT2channels;
    cv::split(_srcFFT, FFT2channels);
    // compute magnitude spectrum of FFT
    cv::magnitude(FFT2channels[0], FFT2channels[1], _dstSpectrum);
}

cv::Mat graySpectrum(cv::Mat spectrum)
{
    cv::Mat dst;
    cv::log(spectrum + 1, dst);
    cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
    // 为了灰度级可视化
    dst.convertTo(dst, CV_8UC1, 255, 0);

    return dst;
}
