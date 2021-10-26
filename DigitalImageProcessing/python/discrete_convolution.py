#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 二维离散卷积(full，valid，same)；可分离卷积核以及其性质
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-17
"""

from scipy import signal
import numpy as np
import cv2 as cv


# --------------------------
if __name__ == "__main__":
    input_matrix = np.array([[1, 2], [3, 4]], np.float32)
    inputH, inputW = input_matrix.shape[:2]
    conv_kernel = np.array([[-1, -2], [2, 1]], np.float32)
    kernelH, kernelW = conv_kernel.shape[:2]

    # 1. 利用 scipy 里面对于信号处理模块里面的卷积操作
    # compute full convolution
    conv_full = signal.convolve2d(
        input_matrix, conv_kernel, mode="full", boundary="fill", fillvalue=0)
    print(f"The result of full convolution: \n{conv_full}")
    # 指定 same 卷积的描点位置
    kernelR, kernelC = 0, 0
    # 根据描点的位置，从 full 卷积中截取得到 same 卷积
    conv_same = conv_full[kernelH - kernelR - 1: inputH + kernelH -
                          kernelR - 1, kernelW - kernelC - 1: inputW + kernelW - kernelC - 1]
    print(f"The result of same convolution: \n{conv_same}")

    # 从 full 卷积中截取得到 valid 卷积
    conv_valid = conv_full[kernelH - 1: inputH, kernelW - 1: kernelW]
    print(f"The result of valid convolution: \n{conv_same}")

    # 2. 利用 OpenCV 里面的函数完成卷积操作
    # 首先对 kernel 矩阵进行翻转 180
    kernelFlip = cv.flip(conv_kernel, -1)
    # 然后进行离散卷积计算
    same_conv = cv.filter2D(input_matrix, -1, kernelFlip,
                            anchor=(0, 0), delta=0.0, borderType=cv.BORDER_CONSTANT)
    print(f"The result of same convolution: \n{same_conv}")

    # 3. 可分离卷积, 进行 zero-padding
    kernel_1 = np.array([[1, 2, 3]], np.float32)
    kernel_2 = np.array([[4], [5], [6]], np.float32)
    # 计算两个卷积的全卷积
    kernel = signal.convolve2d(kernel_1, kernel_2, mode="full")
    print(f"The Separable convolution：\n{kernel}")

    # 4. 利用可分离卷积的性质，减少计算复杂度
    # 针对 full convolution, 两种计算结果相等 first and second.
    inputMatrix = np.array([[1, 2, 3, 10, 12], [32, 43, 12, 4, 190],
                            [12, 234, 78, 0, 12], [43, 90, 32, 8, 90], [71, 12, 4, 98, 123]])
    convKernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    fullConvFirst = signal.convolve2d(inputMatrix, convKernel, mode="full", boundary="fill", fillvalue=0)
    print(f"The first compute way of convolution: \n{fullConvFirst}")
    # convKernel 是可分离的
    convKernel_1 = np.array([[1], [1], [1]], np.float32)
    convKernel_2 = np.array([[1, 0, -1]], np.float32)
    if convKernel.all() != signal.convolve2d(convKernel_1, convKernel_2, mode="full").all():
        print(f"The kernel of convolution is not separable.")
    fullConvSecond_stage = signal.convolve2d(inputMatrix, convKernel_1, mode="full", boundary="fill", fillvalue=0)
    fullConvSecond = signal.convolve2d(fullConvSecond_stage, convKernel_2, mode="full", boundary="fill", fillvalue=0)
    print(f"The second compute way of convolution: \n{fullConvSecond}")

    # 针对 same convolution，两种计算结果相等, 使用 zero-padding
    sameConvFirst = signal.convolve2d(inputMatrix, convKernel, mode="same", boundary="fill", fillvalue=0)
    print(f"The first compute way of convolution: \n{sameConvFirst}")
    # convKernel 是可分离的
    sameConvSecond_stage = signal.convolve2d(inputMatrix, convKernel_1, mode="same", boundary="fill", fillvalue=0)
    sameConvSecond = signal.convolve2d(sameConvSecond_stage, convKernel_2, mode="same", boundary="fill", fillvalue=0)
    print(f"The second compute way of convolution: \n{sameConvSecond}")
    # 如果边界不是 zero-padding，而是其他常数扩充边界，得到的卷积结果不一致，但是只是上下左右边界处不同
    sameWrapConvFirst = signal.convolve2d(inputMatrix, convKernel, mode="same", boundary="wrap", fillvalue=0)
    print(f"The first compute way of convolution: \n{sameWrapConvFirst}")
    # convKernel 是可分离的
    sameWrapConvSecond_stage = signal.convolve2d(inputMatrix, convKernel_1, mode="same", boundary="wrap", fillvalue=0)
    sameWrapConvSecond = signal.convolve2d(sameWrapConvSecond_stage, convKernel_2, mode="same", boundary="wrap", fillvalue=0)
    print(f"The second compute way of convolution: \n{sameWrapConvSecond}")
