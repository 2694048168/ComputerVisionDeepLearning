#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 形态学操作：腐蚀(选择一个任意领域[结构元]里面的最小值)和膨胀(选择一个任意领域[结构元]里面的最大值)
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-11
"""

import sys

import cv2 as cv


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        # 1. 腐蚀操作
        # 创建结构元——矩形
        s = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        # 腐蚀图像，迭代次数为 1
        r = cv.erode(image, s)
        cv.imshow("ErodeImage", r)
        # 边界提取
        e = image - r
        cv.imshow("EdgeImage", e)

        # 2. 膨胀操作
        # 结构元半径
        radio = 1
        MAX_R = 20
        # 显示膨胀效果的窗口
        cv.namedWindow("dilateWin", 1)
        def nothing(*arg):
            pass
        # 调节结构云半径
        cv.createTrackbar("r", "dilateWin", radio, MAX_R, nothing)
        while True:
            # 获取当前的 radio 值
            radio = cv.getTrackbarPos("r", "dilateWin")
            # 创建结构元
            s_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*radio+1, 2*radio+1))
            # 膨胀后图像
            d = cv.dilate(image, s_dilate)
            cv.imshow("DilateImage", d)
            ch = cv.waitKey(5)
            # Esc 退出循环
            if ch == 27:
                break

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
