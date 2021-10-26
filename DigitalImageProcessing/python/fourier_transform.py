#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 二维离散傅里叶变换; 快速傅里叶变换; 幅度谱(零谱中心化)和相位谱
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-21
"""

import sys
import math
import numpy as np
import cv2 as cv


def FFT2Image(src):
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
def AmplitudeSepectrum(img_fft):
    real_part = np.power(img_fft[:, :, 0], 2.0)
    imaginary_part = np.power(img_fft[:, :, 1], 2.0)
    amplitude_part = np.sqrt(real_part + imaginary_part)

    return amplitude_part


# 将计算出的幅度规格化为灰度级显示，幅度矩阵中的值大部分比较大，往往大于 255,
# 如果只是简单地截断为 255 显示，那么幅度谱呈现的信息会很少。
# 一般采用对数函数对幅度谱进行数值压缩，再进行归一化，
# 这样得到的幅度谱的灰度级显示的对比度会比较高。
def graySpectrum(amplitude):
    # 对比度拉伸
    amplitude_log = np.log(amplitude + 1.0)
    # 归一化
    spectrum_norm = np.zeros(amplitude_log.shape, np.float32)
    cv.normalize(amplitude_log, spectrum_norm, 0, 1, cv.NORM_MINMAX)

    return spectrum_norm


# 对于相位谱的计算，利用 Numpy 中的函数 arctan2，
# 该函数的第一个参数是输入的虚部矩阵，
# 第二个参数是输人的实部矩阵，
# 返回值为对应位置的相位角，数值范围为[-pi, pi]，
# 可以将返回的矩阵除以 pi 再乘以 180，规格化到 [-180, 180]
def phaseSpectrum(fft_img):
    rows, cols = fft_img.shape[:2]
    # 计算对应的相位角
    phase_angle = np.arctan2(fft_img[:, :, 1], fft_img[:, :, 0])
    # 将相位角进行转换
    phase_spectrum = phase_angle / math.pi * 180

    return phase_spectrum



# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage", image)

        # ------------ 傅里叶变换 ------------
        img_fft = FFT2Image(image)

        # invert fast Fourier Transform
        img_ifft = np.zeros(img_fft.shape[:2], np.float32)
        cv.dft(img_fft, img_ifft, cv.DFT_INVERSE + cv.DFT_REAL_OUTPUT + cv.DFT_SCALE)

        # 裁剪
        image_ifft = np.copy(img_ifft[:image.shape[0], :image.shape[1]])
        #裁剪后的结果 image_ifft 等于 image, 傅里叶变换是无损操作
        #通过判断原矩阵减去逆变换裁剪后的矩阵是否为零矩阵，验证两者是否相同
        print(f"The operator of Fourier Transform is ?----{np.max(image_ifft - image)}")
        cv.imshow("FFT_Img", image_ifft.astype(np.uint8))

        # ------------ 傅里叶变换中两个度量: 幅度谱和相位谱 ------------
        amplitude_spectrum = AmplitudeSepectrum(img_fft)
        # 幅度谱的灰度级显示
        ampSpectrum = graySpectrum(amplitude_spectrum)
        cv.imshow("AmplitudeSpectrumGrayLevel", ampSpectrum)
        # 相位谱的灰度级显示
        phase_spectrum = phaseSpectrum(img_fft)
        cv.imshow("PhaseSpectrumGrayLevel", phase_spectrum)

        # -------- 傅里叶变换的幅度谱的中心化 零谱的位置(左上角还是中心)--------
        # 仔细观察会发现一个有趣的现象 —— 中心化后的傅里叶谱比较亮的区域大致与原图中的主要目标垂直
        # stpe 1, 矩阵乘以 (-1)^(rows + cols)
        rows, cols = image.shape
        centralization_amplitude_spectrum = np.copy(image)
        centralization_amplitude_spectrum = centralization_amplitude_spectrum.astype(np.float32)
        for row in range(rows):
            for col in range(cols):
                if (row + col) % 2:
                    centralization_amplitude_spectrum[row][col] = -1*image[row][col]
        # step 2, 快速傅里叶变换
        imgFFT = FFT2Image(image)
        # step 3, 计算幅度谱
        amSpec = AmplitudeSepectrum(imgFFT)
        gray_amSpec = graySpectrum(amSpec)
        cv.imshow("GrayAmSpec", gray_amSpec)

        gray_amspec = 255 * gray_amSpec
        cv.imshow("grayImgSpec", gray_amspec.astype(np.uint8))

        cv.waitKey()
        cv.destroyAllWindows()

    else:
        print("Usage: python python-scripy.py imageFile")
