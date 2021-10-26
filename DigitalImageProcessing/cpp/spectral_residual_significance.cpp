/**
 * @File    : spectral_residual_significance.cpp
 * @Brief   : 显著性检测: 谱残差显著性检测
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

        cv::Mat image_float;
        image.convertTo(image_float, CV_64FC1, 1.0 / 255);

        // ------------ 显著性检测: 谱残差检测 ------------
        // step 1, computer fft of image
        cv::Mat fft_img;
        FFT2Image(image_float, fft_img);

        // step 2, compute amplitude spectrum of fft
        cv::Mat amplitude_spectrum;
        AmplitudeSpectrumFFT(fft_img, amplitude_spectrum);
        cv::Mat amplitude_spectrum_log;
        cv::log(amplitude_spectrum + 1.0, amplitude_spectrum_log);

        // step 3, 对幅度谱的灰度级进行均值平滑
        cv::Mat mean_amplitude_spectrum_log;
        cv::blur(amplitude_spectrum_log, mean_amplitude_spectrum_log, cv::Size(3, 3), cv::Point(-1, -1));

        // step 4, 计算谱残差
        cv::Mat spectrum_residual = amplitude_spectrum_log - mean_amplitude_spectrum_log;

        // step 5, 相位谱
        cv::Mat phase_spectrum = PhaseSpectrum(fft_img);
        cv::Mat cos_spectrum(phase_spectrum.size(), CV_64FC1);
        cv::Mat sin_spectrum(phase_spectrum.size(), CV_64FC1);
        for (size_t r = 0; r < phase_spectrum.rows; ++r)
        {
            for (size_t c = 0; c < phase_spectrum.cols; ++c)
            {
                cos_spectrum.at<double>(r, c) = std::cos(phase_spectrum.at<double>(r, c));
                sin_spectrum.at<double>(r, c) = std::sin(phase_spectrum.at<double>(r, c));
            }
        }

        // step 6, 谱残差的幂指数运算
        cv::exp(spectrum_residual, spectrum_residual);
        cv::Mat real_part = spectrum_residual.mul(cos_spectrum);
        cv::Mat imaginary_part = spectrum_residual.mul(sin_spectrum);
        std::vector<cv::Mat> real_imaginary_img;
        real_imaginary_img.push_back(real_part);
        real_imaginary_img.push_back(imaginary_part);
        cv::Mat complex_img;
        cv::merge(real_imaginary_img, complex_img);

        // step 7, 根据新的幅度谱和相位谱, 进行傅里叶逆变换
        cv::Mat ifft_img;
        cv::dft(complex_img, ifft_img, cv::DFT_COMPLEX_OUTPUT + cv::DFT_INVERSE);

        // step 8, 显著性
        cv::Mat ifft_amplitude;
        AmplitudeSpectrumFFT(ifft_img, ifft_amplitude);

        // 平方运算
        cv::pow(ifft_amplitude, 2.0, ifft_amplitude);

        // 对显著性进行高斯平滑
        cv::GaussianBlur(ifft_amplitude, ifft_amplitude, cv::Size(5, 5), 2.5);

        // 显著性显示
        cv::normalize(ifft_amplitude, ifft_amplitude, 1.0, 0, cv::NORM_MINMAX);

        // 利用 伽马变换提高对比度
        cv::pow(ifft_amplitude, 0.5, ifft_amplitude);

        // data type convert
        cv::Mat saliency_map;
        ifft_amplitude.convertTo(saliency_map, CV_8UC1, 255);
        cv::imshow("SaliencyMap", saliency_map);

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
