/**
 * @File    : fourier_transform.cpp
 * @Brief   : 二维离散傅里叶变换; 快速傅里叶变换; 幅度谱(零谱中心化)和相位谱
 * @Author  : Wei Li
 * @Date    : 2021-10-21
*/


#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


// ------------ 快速离散傅里叶变换 ------------
void FFT2Image(cv::InputArray InputImage, cv::OutputArray img_fourier)
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
void AmplitudeSpectrumFFT(cv::InputArray _srcFFT, cv::OutputArray _dstSpectrum)
{
    // 实部和虚部两个通道
    CV_Assert(_srcFFT.channels() == 2);
    // 分离实部和虚部两个通道
    std::vector<cv::Mat> FFT2channels;
    cv::split(_srcFFT, FFT2channels);
    // compute magnitude spectrum of FFT
    cv::magnitude(FFT2channels[0], FFT2channels[1], _dstSpectrum);
}

// 对于傅里叶谱的灰度级显示，OpenCV 提供了函数 log，
// 该函数可以计算矩阵中每一个值的对数。
// 进行归一化后，为了保存傅里叶谱的灰度级，有时需要将矩阵乘以 255，然后转换为 8 位图。
cv::Mat graySpectrum(cv::Mat spectrum)
{
    cv::Mat dst;
    cv::log(spectrum + 1, dst);
    cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
    // 为了灰度级可视化
    dst.convertTo(dst, CV_8UC1,  255, 0);

    return dst;
}

// OpenCV function : cv::phase(x, y, angle, angleInDegress);
cv::Mat PhaseSpectrum(cv::Mat _srcFFT)
{
    cv::Mat phase_spectrum;
    phase_spectrum.create(_srcFFT.size(), CV_64FC1);

    std::vector<cv::Mat> FFT2channels;
    cv::split(_srcFFT, FFT2channels);

    // 计算相位谱
    for (size_t r = 0; r < phase_spectrum.rows; ++r)
    {
        for (size_t c = 0; c < phase_spectrum.cols; ++c)
        {
            double real_part = FFT2channels[0].at<double>(r, c);
            double imaginary_part = FFT2channels[1].at<double>(r, c);
            // atan2 返回值范围 [0, 180], [-180, 0]
            phase_spectrum.at<double>(r, c) = std::atan2(imaginary_part, real_part);
        }
    }

    return phase_spectrum;
}


// -------------------------------
int main(int argc, char** argv)
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

        // ------------ 离散傅里叶变换 ------------
        // 输入类型转换 CV_8U ----> CV_32F or CV_64F
        cv::Mat image_float;
        image.convertTo(image_float, CV_64F);

        // Fourier Transform
        cv::Mat img_fourier;
        cv::dft(image_float, img_fourier, cv::DFT_COMPLEX_OUTPUT);
        // Invert Fourier Transform, only real
        cv::Mat img_invert_fourier;
        cv::dft(img_fourier, img_invert_fourier, cv::DFT_REAL_OUTPUT + cv::DFT_INVERSE + cv::DFT_SCALE);

        // convert float dataType to int dataType
        cv::Mat image_int;
        img_invert_fourier.convertTo(image_int, CV_8U);
        cv::imshow("FourierImg", image_int);

        // ------------ 快速离散傅里叶变换 ------------
        cv::Mat img_fft;
        FFT2Image(image_float, img_fft);

        cv::Mat img_ifft;
        cv::dft(img_fft, img_ifft, cv::DFT_REAL_OUTPUT + cv::DFT_INVERSE + cv::DFT_SCALE);
        // 通过裁剪傅里叶逆变换的实部获取的结果 等同原来的输入图像
        cv::Mat img = img_ifft(cv::Rect(0, 0, image.cols, image.rows)).clone();

        img_ifft.convertTo(img_ifft, CV_8U);
        cv::imshow("FFT_Img", img_ifft);

        // ------------ 傅里叶变换的两个度量: 幅度谱和相位谱 ------------
        cv::Mat amplitude_spectrum;
        AmplitudeSpectrumFFT(img_fft, amplitude_spectrum);
        amplitude_spectrum = graySpectrum(amplitude_spectrum);
        cv::imshow("AmplitudeSpectrum", amplitude_spectrum);

        cv::Mat phase_spectrum = PhaseSpectrum(img_fft);
        cv::imshow("PhaseSpectrum", phase_spectrum);

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
