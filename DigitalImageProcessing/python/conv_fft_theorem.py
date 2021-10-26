#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 卷积定理: 卷积定义和傅里叶变换的关系(利用快速傅里叶变换)
@     对于卷积核为任意尺寸或者锚点在任意位置的情况，只是最后的裁剪部分不同。
@    虽然通过定义计算卷积比较耗时，但是当卷积核较小时，通过快速傅里叶变换计算卷积并没有明显的优势;
@    只有当卷积核较大时，利用傅里叶变换的快速算法计算卷积才会表现出明显的优势。
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-21
"""

from scipy import signal
import numpy as np
import cv2 as cv


def FFT2Conv(image, kernel, _borderType=cv.BORDER_DEFAULT):
    """利用快速傅里叶变换完成卷积的计算.

    Args:
        image ([ndarray]): 输入图像
        kernel ([ndarray]): 卷积核
        borderType ([type], optional): 边界填充方式. Defaults to cv.BORDER_DEFAULT.

    Returns:
        [ndarray]: smae convolution 
    """
    R, C = image.shape[:2]
    r, c = kernel.shape[:2]
    tb = (r - 1) / 2
    lr = (c - 1) / 2

    img_padding = cv.copyMakeBorder(image, tb, tb, lr, lr, _borderType)
    rows = cv.getOptimalDFTSize(img_padding.shape[0] + r - 1)
    cols = cv.getOptimalDFTSize(img_padding.shape[1] + c - 1)
    img_zero_padding = np.zeros((rows, cols), np.float64)
    img_zero_padding[:img_padding.shape[0], :img_padding.shape[1]] = img_padding

    kernel_zeros = np.zeros((rows, cols), np.float64)
    kernel_zeros[:kernel.shape[0], :kernel.shape[1]] = kernel

    fft_ipz = np.zeros((rows, cols, 2), np.float64)
    cv.dft(img_zero_padding, fft_ipz, cv.DFT_COMPLEX_OUTPUT)
    fft_kz = np.zeros((rows, cols, 2), np.float64)
    cv.dft(kernel_zeros, fft_kz, cv.DFT_COMPLEX_OUTPUT)

    ipz_rz = cv.mulSpectrums(fft_ipz, fft_kz, cv.DFT_ROWS)

    ifft2FullConv = np.zeros((rows, cols), np.float64)
    cv.dft(ipz_rz, ifft2FullConv, cv.DFT_INVERSE + cv.DFT_SCALE + cv.DFT_REAL_OUTPUT)
    print(f"{np.max(ifft2FullConv)}")

    sameConv = np.copy(ifft2FullConv[r - 1 : R + r - 1, c - 1 : C + c - 1])

    return sameConv


# --------------------------
if __name__ == "__main__":
    pass