/**
 * @File    : custom_filter.cpp
 * @Brief   : 自定义频域滤波器，消除指定结构或者目标
 * @Author  : Wei Li
 * @Date    : 2021-10-22
*/


#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat image;     // 输入的图像矩阵
cv::Mat fImageFFT; // 图像的快速傅里叶变换
cv::Point maxLoc;  // 傅里叶谱的最大值的坐标
// ------------ 快速离散傅里叶变换 ------------
cv::String windowName = "AmplitudeSpectrum";
cv::Mat fImageFFT_spectrum; // 傅里叶变换的傅里叶谱
bool drawing_box = false;   // 鼠标事件
cv::Point downPoint;
cv::Rect rectFilter;
bool gotRectFilter = false;
void mouseRectHandler(int event, int x, int y, int, void *);

// 快速傅里叶变换
void fft2Image(cv::InputArray _src, cv::OutputArray _dst);
// 幅度谱
void amplitudeSpectrum(cv::InputArray _srcFFT, cv::OutputArray _dstSpectrum);
// 幅度谱的灰度级显示
cv::Mat graySpectrum(cv::Mat spectrum);

// --------------------------------------------------------
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
        cv::Mat fImage;
        image.convertTo(fImage, CV_32FC1, 1.0, 0.0);

        // ------------ step 2, each pixel value multiply (-1)^(r+c) ------------
        for (size_t r = 0; r < fImage.rows; ++r)
        {
            for (size_t c = 0; c < fImage.cols; ++c)
            {
                if ((r + c) % 2)
                {
                    fImage.at<float>(r, c) *= -1;
                }
            }
        }

        // ------------ step 3 and step 4, zero-padding and FFT ------------
        fft2Image(fImage, fImageFFT);
        cv::Mat amplSpec;
        amplitudeSpectrum(fImageFFT, amplSpec);
        cv::Mat spectrum = graySpectrum(amplSpec);
        cv::minMaxLoc(spectrum, nullptr, nullptr, nullptr, &maxLoc);

        // ------------ step 5, 自定义频率滤波器，消除指定结构或者目标 ------------
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(windowName, mouseRectHandler, nullptr);
        while (true)
        {
            spectrum(rectFilter).setTo(0);

            // ------------ step 6, 自定义滤波器和傅里叶变换进行点乘操作 ------------
            fImageFFT(rectFilter).setTo(cv::Scalar::all(0));
            cv::imshow(windowName, spectrum);
            // ESC 退出编辑
            if (cv::waitKey(10) == 27)
            {
                break;
            }
        }

        // ------------ step 7 and step 8, 傅里叶逆变换，取实部 ------------
        cv::Mat result;
        cv::dft(fImageFFT, result, cv::DFT_SCALE + cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT);

        // ------------ step 9, (-1)^(r+c) ------------
        int rows = result.rows;
        int cols = result.cols;
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                if ((r + c) % 2)
                {
                    result.at<float>(r, c) *= -1;
                }
            }
        }

        // ------------ step 10, 裁剪矩阵，取左上角部分，大小与输入一致 ------------
        result.convertTo(result, CV_8UC1, 1.0, 0);
        result = result(cv::Rect(0, 0, image.cols, image.rows));
        cv::imshow("FilterImage", result);

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

// 鼠标事件
void mouseRectHandler(int event, int x, int y, int, void *)
{
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN: // 鼠标左键
        drawing_box = true;
        // 记录起点
        downPoint = cv::Point(x, y);
        break;

    case cv::EVENT_MOUSEMOVE: // 鼠标移动
        if (drawing_box)
        {
            // 将鼠标指针移动到 downPoint 的右下角
            if (x >= downPoint.x && y >= downPoint.y)
            {
                rectFilter.x = downPoint.x;
                rectFilter.y = downPoint.y;
                rectFilter.width = x - downPoint.x;
                rectFilter.height = y - downPoint.y;
            }

            // 将鼠标指针移动到 downPoint 的右上角
            if (x >= downPoint.x && y <= downPoint.y)
            {
                rectFilter.x = downPoint.x;
                rectFilter.y = y;
                rectFilter.width = x - downPoint.x;
                rectFilter.height = downPoint.y - y;
            }

            // 将鼠标指针移动到 downPoint 的左上角
            if (x <= downPoint.x && y <= downPoint.y)
            {
                rectFilter.x = x;
                rectFilter.y = y;
                rectFilter.width = downPoint.x - x;
                rectFilter.height = downPoint.y - y;
            }

            // 将鼠标指针移动到 downPoint 的左下角
            if (x <= downPoint.x && y >= downPoint.y)
            {
                rectFilter.x = x;
                rectFilter.y = downPoint.y;
                rectFilter.width = downPoint.x - x;
                rectFilter.height = y - downPoint.y;
            }
        }
        break;

    case cv::EVENT_LBUTTONUP: // 鼠标左键松开
        drawing_box = false;
        gotRectFilter = true;
        break;

    default:
        break;
    }
}

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
