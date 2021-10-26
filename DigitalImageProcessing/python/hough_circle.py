#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 霍夫圆检测；基于梯度的霍夫圆检测
         尽管标准的霍夫变换对于曲线检测是一项强有力的技术，但是随着曲线参数数目的增加，造成计数器的数据结构越来越复杂，如直线检测的计数器是二维的，圆检测的计数器是三维的，这需要大量的存储空间和巨大的计算量，因此通常采用其他方法进行改进，如同概率直线检测对标准霍夫直线检测的改进,那么基于梯度的霍夫圆检测就是对标准霍夫圆检测的改进
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-18
"""

import sys
import math
import numpy as np
import cv2 as cv


def HoughTransformCircle(image, minR, maxR, voteThresh=100):
    H, W = image.shape
    minr = round(minR) + 1
    maxr = round(maxR) + 1
    r_num = int(maxr - minr + 1)
    a_num = int(W - 1 + maxr + maxr + 1)
    b_num = int(H - 1 + maxr + maxr + 1)
    accumulator = np.zeros((r_num, b_num, a_num), np.int32)
    # 投票计数
    for y in range(H):
        for x in range(W):
            if (image[y][x] == 255):
                for k in range(r_num):
                    for theta in np.linspace(0, 360, 180):
                        # 计算对应的 a 和 b
                        a = x - (minr + k) * math.cos(theta/180.0 * math.pi)
                        b = y - (minr + k) * math.sin(theta/180.0 * math.pi)
                        a = int(round(a))
                        b = int(round(b))
                        accumulator[k, b, a] += 1

    # 筛选投票数大于 voteThresh 的圆
    circles = []
    for k in range(r_num):
        for b in range(b_num):
            for a in range(a_num):
                if (accumulator[k, b, a] > voteThresh):
                    circles.append((k+minr, b, a))

    return circles

# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        # cv.imshow("OriginImage", image)

        # 边缘二值图
        edge_canny = cv.Canny(image, 50, 200)
        cv.imshow("EdgeCanny", edge_canny)

        # -------- Hough Transform Circle --------
        circles = HoughTransformCircle(edge_canny, 40, 50, 80)
        # 可视化检测结果
        for i in range(len(circles)):
            cv.circle(image, (int(circles[i][2]), int(circles[i][1])), int(circles[i][0]), (255), 2)
        cv.imshow("CircleImage", image)

        # 基于梯度的霍夫圆检测
        hough_circle = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 100, 200, param2=60, minRadius=54)
        print(type(hough_circle))
        print(hough_circle.shape)
        for i in range(hough_circle.shape[1]):
            circle_center = (int(hough_circle[0, i, 0]), int(hough_circle[0, i, 1]))
            circle_radius = hough_circle[0, i, 2]
            cv.circle(image, circle_center, circle_radius, 255, 3)

        cv.imshow("src", image)

        cv.waitKey()
        cv.destroyAllWindows()

    else:
        print("Usage: python hough_transform.py imageFile")
