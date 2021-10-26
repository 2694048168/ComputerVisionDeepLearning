#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 形态学操作：开运算(腐蚀后膨胀)和闭运算(膨胀后腐蚀); 顶帽变换、底帽变换、形态学梯度
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-12
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

        # 结构元半径和迭代次数以及最大值
        r, i = 1, 1
        MAX_R, MAX_I = 20, 20

        # 显示形态学处理效果的窗口
        cv.namedWindow("morphology", 1)
        # 回调函数
        def nothing(*arg):
            pass
        # 调节结构元半径
        cv.createTrackbar("r", "morphology", r, MAX_R, nothing)
        # 调节迭代次数
        cv.createTrackbar("i", "morphology", r, MAX_I, nothing)

        while True:
            # 获取当前进度条上的 r 半径值
            r = cv.getTrackbarPos("r", "morphology")
            # 获取当前进度条上的 i 迭代次数
            i = cv.getTrackbarPos("i", "morphology")
            # 创建结构元
            s = cv.getStructuringElement(cv.MORPH_RECT, (2*r+1, 2*r+1))

            # 1. 进行形态学处理——开运算
            # d = cv.morphologyEx(image, cv.MORPH_OPEN, s, iterations=i)

            # 2. 进行形态学处理——闭运算
            # d = cv.morphologyEx(image, cv.MORPH_CLOSE, s, iterations=i)

            # 3. 进行形态学处理——顶帽变换
            # d = cv.morphologyEx(image, cv.MORPH_TOPHAT, s, iterations=i)

            # 4. 进行形态学处理——底帽变换
            # d = cv.morphologyEx(image, cv.MORPH_BLACKHAT, s, iterations=i)

            # 5. 进行形态学处理——形态学梯度
            d = cv.morphologyEx(image, cv.MORPH_GRADIENT, s, iterations=i)

            # 可视化处理效果
            cv.imshow("morphology", d)
            # cv.imwrite("./image/open.png", d)
            ch = cv.waitKey(5)
            # Esc 退出循环
            if ch == 27:
                break

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
