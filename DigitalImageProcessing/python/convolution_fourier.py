#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 卷积定理: 卷积定义和傅里叶变换的关系
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-21
"""

from scipy import signal
import numpy as np
import cv2 as cv


# --------------------------
if __name__ == "__main__":
    image_matrix = np.array([[34, 56, 1, 0, 255, 230, 45, 12], [0, 201, 101, 125, 52, 12, 124, 12],
                            [3, 41, 42, 40, 12, 90, 123, 45], [
                                5, 245, 98, 32, 34, 234, 90, 123],
                            [12, 12, 10, 41, 56, 89, 189, 5], [
                                112, 87, 12, 45, 78, 45, 10, 1],
                            [42, 123, 234, 12, 12, 21, 56, 43], [1, 2, 45, 123, 10, 44, 123, 90]], np.float64)

    kernel = np.array([[1, 0, -1], [1, 0, 1], [1, 0, -1]], np.float64)

    # ------------- 卷积定理 -------------
    conv_full = signal.convolve2d(image_matrix, kernel, mode="full", boundary="fill", fillvalue=0)

    ft_img = np.zeros((image_matrix.shape[0], image_matrix.shape[1], 2), np.float64)
    cv.dft(image_matrix, ft_img, cv.DFT_COMPLEX_OUTPUT)
    ft_kernel = np.zeros((kernel.shape[0], kernel.shape[1], 2), np.float64)
    cv.dft(kernel, ft_kernel, cv.DFT_COMPLEX_OUTPUT)

    fft_img = np.zeros((conv_full.shape[0], conv_full.shape[1]), np.float64)
    # image 右侧和下侧 zero-padding
    img_padding = np.zeros((image_matrix.shape[0]+kernel.shape[0] - 1, image_matrix.shape[0]+kernel.shape[1] - 1), np.float64)
    img_padding[:image_matrix.shape[0], :image_matrix.shape[1]] = image_matrix
    ft_img_padding = np.zeros((img_padding.shape[0], img_padding.shape[1], 2), np.float64)
    cv.dft(img_padding, ft_img_padding, cv.DFT_COMPLEX_OUTPUT)

    # kernel 右侧和下侧 zero-padding
    kernel_padding = np.zeros((image_matrix.shape[0]+kernel.shape[0]-1, image_matrix.shape[1]+kernel.shape[1]-1), np.float64)
    kernel_padding[:kernel.shape[1], :kernel.shape[1]] = kernel
    ft_kernel_padding = np.zeros((kernel_padding.shape[0], kernel_padding.shape[1], 2), np.float64)
    cv.dft(kernel_padding, ft_kernel_padding, cv.DFT_COMPLEX_OUTPUT)

    # 两个傅里叶变换对应位置相乘
    ft_img_kernel = cv.mulSpectrums(ft_img_padding, ft_kernel_padding, cv.DFT_ROWS)

    # 利用 傅里叶变换计算 full 卷积
    ifft_img = np.zeros(ft_img_kernel.shape[:2], np.float64)
    cv.dft(ft_img_kernel, ifft_img, cv.DFT_INVERSE + cv.DFT_SCALE + cv.DFT_REAL_OUTPUT)

    # 通过两个方面来理解卷积定理:
    # 第一，根据卷积定义计算出的 image和 kernel的卷积结果，是否与通过傅里叶变换计算出的卷积结果相同;
    print(f"{np.max(ifft_img - conv_full)}")

    # 第二，对于 image 和 kernel 的全卷积结果的傅里叶变换，是否与两个核扩充后的傅里叶变换点乘相同。
    FT_convfull = np.zeros((conv_full.shape[0], conv_full.shape[1], 2), np.float64)
    cv.dft(conv_full, FT_convfull, cv.DFT_COMPLEX_OUTPUT)
    print(f"{np.max(FT_convfull - ft_img_kernel)}")
