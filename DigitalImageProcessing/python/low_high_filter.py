#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 低通滤波器和高通滤波器(理想滤波器；巴特沃斯滤波器；高斯滤波器)
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-22
"""


import sys
import numpy as np
import cv2 as cv


def fft2Image(src):
    rows, cols = src.shape[:2]
    # 获取快速傅里叶变换的最优扩充
    row_padding = cv.getOptimalDFTSize(rows)
    col_padding = cv.getOptimalDFTSize(cols)

    # 下侧面和右侧面进行 zero-padding
    img_fft = np.zeros((row_padding, col_padding, 2), np.float32)
    img_fft[:rows, :cols, 0] = src

    # 快速傅里叶变换
    cv.dft(img_fft, img_fft, cv.DFT_COMPLEX_OUTPUT)
    return img_fft


# ------------ 傅里叶变换中两个度量: 幅度谱和相位谱 ------------
def amplitudeSepectrum(img_fft):
    real_part = np.power(img_fft[:, :, 0], 2.0)
    imaginary_part = np.power(img_fft[:, :, 1], 2.0)
    amplitude_part = np.sqrt(real_part + imaginary_part)

    return amplitude_part


def graySpectrum(amplitude):
    # 对比度拉伸
    amplitude_log = np.log(amplitude + 1.0)
    # 归一化
    spectrum_norm = np.zeros(amplitude_log.shape, np.float32)
    cv.normalize(amplitude_log, spectrum_norm, 0, 1, cv.NORM_MINMAX)

    return spectrum_norm


def createLPFilter(shape, center, radius, lpType=2, n=2):
    """构建三种低通滤波器：理想滤波器，巴特沃斯滤波器，高斯滤波器

    Args:
        shape ([tuple]): 滤波器的大小，表示快速傅里叶变换的尺寸; (high, width)
        center ([tuple]): 傅里叶谱的中心位置; (x, y)
        radius ([float]): 截至频率；
        lpType (int, optional): 滤波器的类型. Defaults to 2.
        n (int, optional): 巴特沃斯滤波器的阶数. Defaults to 2.

    Returns:
        [ndarray]: 低通滤波器
    """
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c = c - center[0]
    r = r - center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)

    lpFilter = np.zeros(shape, np.float32)
    if radius <= 0:
        return lpFilter

    # case 1, 理想低通滤波器
    if lpType == 0:
        lpFilter = np.copy(d)
        lpFilter[lpFilter < pow(radius, 2.0)] = 1
        lpFilter[lpFilter >= pow(radius, 2.0)] = 0
    # case 2, 巴特沃斯低通滤波器
    elif lpType == 1:
        lpFilter = 1.0 / (1.0 + np.power(np.sqrt(d) / radius, 2 * n))
    # case 3, 高斯低通滤波器
    elif lpType == 2:
        lpFilter = np.exp(-d / (2.0 * pow(radius, 2.0)))

    return lpFilter


# --------------------------
# 截至频率
radius = 50
MAX_RADIUS = 100
lpType = 0
MAX_LPTYPE = 2
# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or directory.")
            sys.exit()
        cv.imshow("OriginImage", image)

        # ------------ step 1, reading image file ------------
        fimage = np.zeros(image.shape, np.float)

        # ------------ step 2, (-1)^(r+c) ------------
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                if (r+c) % 2:
                    fimage[r][c] = -1 * image[r][c]
                else:
                    fimage[r][c] = image[r][c]

        # ------------ step 3 and step 4, zero-padding and FFT ------------
        FImagefft2 = fft2Image(fimage)
        amplitude = amplitudeSepectrum(FImagefft2)
        spectrum = graySpectrum(amplitude)
        cv.imshow("OriginalSpectrum", spectrum)

        minValue, maxValue, minLoc, maxLoc = cv.minMaxLoc(amplitude)
        cv.namedWindow("lpFilterSpectrum", 1)
        def nothing(*arg):
            pass
        
        cv.createTrackbar("lpType", "lpFilterSpectrum", lpType, MAX_LPTYPE, nothing)
        cv.createTrackbar("radius", "lpFilterSpectrum", radius, MAX_RADIUS, nothing)
        result = np.zeros(spectrum.shape, np.float32)
        while True:
            radius = cv.getTrackbarPos("radius", "lpFilterSpectrum")
            lpType = cv.getTrackbarPos("lpType", "lpFilterSpectrum")
            # ------------ step 5, 构建低通滤波器 ------------
            lpFilter = createLPFilter(spectrum.shape, maxLoc, radius, lpType)

            # ------------ step 6, 低通滤波器和快速傅里叶变换的对应位置做点乘 ------------
            rows, cols = spectrum.shape[:2]
            fImagefft2_lpFilter = np.zeros(FImagefft2.shape, FImagefft2.dtype)
            for i in range(2):
                fImagefft2_lpFilter[:rows, :cols, i] = FImagefft2[:rows, :cols, i] * lpFilter

            lp_amplitude = amplitudeSepectrum(fImagefft2_lpFilter)
            lp_spectrum = graySpectrum(lp_amplitude)
            cv.imshow("lpFilterSpectrum", lp_spectrum)

            # ------------ step 7 and step 8, 对低通滤波器变换执行傅里叶逆变换，取实部 ------------
            cv.dft(fImagefft2_lpFilter, result, cv.DFT_REAL_OUTPUT + cv.DFT_INVERSE + cv.DFT_SCALE)

            # ------------ step 9, (-1)^(r+c) ------------
            for r in range(rows):
                for c in range(cols):
                    if (r + c) % 2:
                        result[r][c] = -1 * result[r][c]

            # ------------ step 10, 数据类型转换，截取左方角部分 ------------
            for r in range(rows):
                for c in range(cols):
                    if result[r][c] < 0:
                        result[r][c] = 0
                    elif result[r][c] > 255:
                        result[r][c] = 255
            lpResult = result.astype(np.uint8)
            lpResult = lpResult[:image.shape[0], :image.shape[1]]
            cv.imshow("LPFilter", lpResult)
            ch = cv.waitKey(5)
            if ch == 27:
                break

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python python-scripy.py imageFile.")
