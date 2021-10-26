#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 显著性检测: 谱残差显著性检测
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


def AmplitudeSepectrum(img_fft):
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

        # ------------ 显著性检测: 谱残差检测 ------------
        # step 1, computer fft of image
        fft_img = FFT2Image(image)

        # step 2, compute amplitude spectrum of fft
        amplitude_spectrum = AmplitudeSepectrum(fft_img)
        # compute gray level of amplitude spectrum
        amplitude_spectrum_log = graySpectrum(amplitude_spectrum)

        # step 3, compute phase spectrum of fft
        phase_spectrum = phaseSpectrum(fft_img)
        # 利用余弦函数计算余弦谱, 对应实部
        cos_phase_spectrum = np.cos(phase_spectrum)
        # 利用正弦函数计算正弦谱, 对应虚部
        sin_phase_spectrum = np.sin(phase_spectrum)

        # step 4, 对幅度谱的灰度级进行均值平滑
        mean_log_amplitude_spectrum = cv.boxFilter(amplitude_spectrum_log, cv.CV_32FC1, (3, 3))

        # step 5, 计算谱残差
        amplitude_spectrum_residual = amplitude_spectrum_log - mean_log_amplitude_spectrum

        # step 6, 谱残差的幂指数运算
        exp_amplitude_spectrum_residual = np.exp(amplitude_spectrum_residual)

        # 分别计算实数部分和虚数部分
        real_part = exp_amplitude_spectrum_residual * cos_phase_spectrum
        imaginary_part = exp_amplitude_spectrum_residual * sin_phase_spectrum
        # 合并实部和虚部
        com_real_imaginary = np.zeros((real_part.shape[0], real_part.shape[1], 2), np.float32)
        com_real_imaginary[:, :, 0] = real_part
        com_real_imaginary[:, :, 1] = imaginary_part

        # step 7, 根据新的幅度谱和相位谱, 进行傅里叶逆变换
        ifft_img = np.zeros(com_real_imaginary.shape, np.float32)
        cv.dft(com_real_imaginary, ifft_img, cv.DFT_COMPLEX_OUTPUT + cv.DFT_INVERSE)

        # step 8, 显著性
        saliency_map = np.power(ifft_img[:, :, 0], 2) + np.power(ifft_img[:, :, 1], 2)

        # 对显著性进行高斯平滑
        saliency_map = cv.GaussianBlur(saliency_map, (5, 5), 2.5)
        # show the saliency map for test
        # saliency_map = cv.normalize(saliency_map, saliency_map, 0, 1, cv.NORM_MINMAX)
        saliency_map = saliency_map / np.max(saliency_map)

        # 利用 伽马变换提高对比度
        saliency_map = np.power(saliency_map, 0.5)

        saliency_map = np.round(saliency_map*255)
        saliency_map = saliency_map.astype(np.uint8)
        cv.imshow("SaliencyMap", saliency_map)

        cv.waitKey()
        cv.destroyAllWindows()

    else:
        print("Usage: python python-scripy.py imageFile")
