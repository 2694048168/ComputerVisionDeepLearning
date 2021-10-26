#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 利用极坐标变换对图像进行变换，校正图像中的圆形区域
OpenCV 实现了线性极坐标变换和对数极坐标变换。
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-15
"""

import sys
import numpy as np
import cv2 as cv


def polar(input_img, center, r, theta=(0, 360), r_step=1.0, theta_step=360.0/(180*8)):
    """实现图像的极坐标变换。

    Args:
        input_img (ndarray): 输入图像
        center ([type]): 极坐标变换中心
        r (二元元组 tuple): 最小距离和最大距离
        theta (tuple, optional): 角度范围. Defaults to (0, 360).
        r_step (float, optional): r 极长进行离散化的步长. Defaults to 1.0.
        theta_step ([type], optional): theta 角度进行离散化的步长. Defaults to 360.0/(180*8).

    Returns:
        [ndarray]: 输出图像
    """
    min_r, max_r = r
    min_theta, max_theta = theta
    cx, cy = center
    # 输出图像的 H W
    H = int((max_r - min_r) / r_step) + 1
    W = int((max_theta - min_theta) / theta_step) + 1
    output_img = 125 * np.ones((H, W), input_img.dtype)

    # 极坐标变换
    r = np.linspace(min_r, max_r, H)
    r = np.tile(r, (W, 1))
    r = np.transpose(r)
    theta = np.linspace(min_theta, max_theta, W)
    theta = np.tile(theta, (H, 1))
    x, y = cv.polarToCart(r, theta, angleInDegrees=True)

    # 插值算法 最邻近插值
    for i in range(H):
        for j in range(W):
            px = int(round(x[i][j] + cx))
            py = int(round(y[i][j] + cy))
            if (px>=0 and px<=W-1) and (py>=0 and py<=H-1):
                output_img[i][j] = input_img[i][j] 

    return output_img


# --------------------------
if __name__ == "__main__":
    # 0. tips for coding with Numpy
    # The function tile() of numpy.
    a = np.array([[1, 2], [3, 4]])
    # 将 a 分别再垂直方向和水平方向复制 2 次和 3 次
    b = np.tile(a, (2, 3))
    print(f"The function tile of Numpy: \n{b}")

    # 1. 极坐标变换对图像
    if len(sys.argv) > 1:
        input_img = cv.imread(sys.argv[1], 0)
        h, w = input_img.shape[:2]
        cx, cy = 508, 503
        cv.circle(input_img, (int(cx), int(cy)), 10, (255.0, 0, 0), 3)
        output_img = polar(input_img=input_img, center=(cx, cy), r=(200, 550))
        # 翻转
        output_img = cv.flip(output_img, 0)

        cv.imshow("OriginImage", input_img)
        cv.imshow("OutputImage", output_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python polar image.")