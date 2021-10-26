#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 同态滤波
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


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # ------------ step 1, reading image file ------------
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or directory.")
            sys.exit()
        cv.imshow("OriginImage", image)

        # ------------ step 2, 对数操作处理 ------------
        image_log = np.log(image + 1.0)
        image_log = image_log.astype(np.float32)

        # ------------ step 3, (-1)^(r+c) ------------
        image_fourier = np.copy(image_log)
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                if (r + c) % 2:
                    image_fourier[r][c] = -1 * image_fourier[r][c]

        # ------------ step 4 and step 5, zero-padding and FFT ------------
        fft2 = fft2Image(image_fourier)

        # ------------ step 6, High-Emphasis Filter, 高频增强滤波器 ------------
        amplitude = amplitudeSepectrum(fft2)
        minValue, maxValue, minLoc, maxLoc = cv.minMaxLoc(amplitude)
        rows, cols = fft2.shape[:2]
        r, c = np.mgrid[0:rows:1, 0:cols:1]
        c = c - maxLoc[0]
        r = r - maxLoc[1]
        d = np.power(c, 2.0) + np.power(r, 2.0)
        high, low, k, radius = 2.5, 0.5, 1, 300
        heFilter = (high - low) * (1 - np.exp(-k * d / (2.0 * pow(radius, 2.0)))) + low

        # ------------ step 7, 快速傅里叶变换与高频增强滤波的点乘 ------------
        fft2Filter = np.zeros(fft2.shape, fft2.dtype)
        for i in range(2):
            fft2Filter[:rows, :cols, i] = fft2[:rows, :cols, i] * heFilter

        # ------------ step 8 and step 9, 高频增强傅里叶逆变换，取实部 ------------
        ifft2 = cv.dft(fft2Filter, flags=cv.DFT_REAL_OUTPUT+cv.DFT_INVERSE+cv.DFT_SCALE)

        # ------------ step 10, 裁剪，使其大小与原始图像一致 ------------
        ifftImage = np.copy(ifft2[:image.shape[0], :image.shape[1]])

        # ------------ step 11, (-1)^(r+c) ------------
        for i in range(ifftImage.shape[0]):
            for j in range(ifftImage.shape[1]):
                if (i + j) % 2:
                    ifftImage[i][j] = -1 * ifftImage[i][j]

        # ------------ step 12,  对数的反操作：取指数 ------------
        exp_img = np.exp(ifftImage) - 1

        # ------------ step 13,  归一化 ------------
        exp_img = (exp_img - np.min(exp_img)) / (np.max(exp_img) - np.min(exp_img))
        exp_img = 255 * exp_img
        exp_img = exp_img.astype(np.uint8)
        cv.imshow("HomomorphicFilter", exp_img)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python python-scripy.py imageFile.")
