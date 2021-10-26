#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 霍夫变换 (Hough Transform) 进行二值图像的直线检测
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-18
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv


def HoughTransformLine(image, stepTheta=1, stepRho=1):
    """实现标准的霍夫直线检测

    Args:
        image ([ndarray]): 输入待检测的二值图像
        stepTheta (int, optional): 计算机程序实现原理时候的离散化参数 θ. Defaults to 1.
        stepRho (int, optional): 计算机程序实现原理时候的离散化参数 ρ. Defaults to 1.

    Returns:
        [tuple]: (计数器，对应的哪些点是共线的)
    """
    rows, cols = image.shape
    # 图像中可能出现的最大垂线的长度
    L = round(math.sqrt(pow(rows - 1, 2.0) + pow(cols - 1, 2.0))) + 1
    # 初始化投票器
    numTheta = int(180.0 / stepTheta)
    numRho = int(2 * L / stepRho + 1)
    accumulator = np.zeros((numRho, numTheta), np.int32)
    # 建立字典, 对应的那些点是共线的
    accuDict = {}
    for k1 in range(numRho):
        for k2 in range(numTheta):
            accuDict[(k1, k2)] = []

    # 投票计数
    for y in range(rows):
        for x in range(cols):
            if (image[y][x] == 255):  # 只对边缘点做霍夫变换
                for m in range(numTheta):
                    # 对每一个角度，计算对应的 rho 值
                    rho = x * math.cos(stepTheta * m / 180.0 * math.pi) + y * math.sin(stepTheta * m / 180.0 * math.pi)
                    # 计算投票哪一个区域
                    n = int(round(rho + L) / stepRho)
                    # 投票计数 +1
                    accumulator[n, m] += 1
                    # 记录该点
                    accuDict[(n, m)].append((y, x))

    return accumulator, accuDict 

# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage", image)

        # 边缘二值图
        edge_canny = cv.Canny(image, 50, 200)
        cv.imshow("EdgeCanny", edge_canny)

        # -------- Hough Transform Line --------
        # -------- 对计数器进行三维展示和二值化展示 --------
        accumulator, accuDict = HoughTransformLine(edge_canny, 1, 1)

        # 计算器的二维直方图展示
        rows, cols = accumulator.shape
        fig = plt.figure()
        plt.gca(projection="3d")
        ax = fig.add_subplot(projection='3d')
        X, Y = np.mgrid[0:rows:1, 0:cols:1]
        surface = ax.plot_wireframe(X, Y, accumulator, cstride=1, rstride=1, color="gray")
        ax.set_xlabel(u"$\\rho$")
        ax.set_ylabel(u"$\\theta$")
        ax.set_zlabel("accumulator")
        ax.set_zlim3d(0, np.max(accumulator))
        plt.show()
        plt.close()

        # 计数器的灰度级显示
        grayAccu = accumulator / float(np.max(accumulator))
        grayAccu = 255 * grayAccu
        grayAccu = grayAccu.astype(np.uint8)
        # 只需要绘制投票数大于 60 的直线
        voteThresh = 60
        for r in range(rows):
            for c in range(cols):
                if accumulator[r][c] > voteThresh:
                    points = accuDict[(r, c)]
                    cv.line(image, points[0], points[len(points) - 1], (255), 2)
        cv.imshow("Accumulator", grayAccu)

        cv.waitKey()
        cv.destroyAllWindows()

    else:
        print("Usage: python hough_transform.py imageFile")
