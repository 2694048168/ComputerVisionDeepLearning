#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 色彩空间的转换; 调整彩色图像的饱和度和亮度
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-21
"""


import sys
import numpy as np
import cv2 as cv


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1])
        if image is None:
            print(f"Error: no such file or directory.")
            sys.exit()
        cv.imshow("OriginImage", image)

        img_float = image.astype(np.float32)
        image = img_float / 255.0
        HLS_Img = cv.cvtColor(image, cv.COLOR_BGR2HLS)

        l_portion = 0
        s_portion = 0
        MAX_VALUE = 100
        cv.namedWindow("l and s", cv.WINDOW_AUTOSIZE)
        def nothing(*arg):
            pass
        cv.createTrackbar("l", "l and s", l_portion, MAX_VALUE, nothing)
        cv.createTrackbar("s", "l and s", s_portion, MAX_VALUE, nothing)

        lsImg = np.zeros(image.shape, np.float32)
        while True:
            hlsCopy = np.copy(HLS_Img)
            l_portion = cv.getTrackbarPos("l", "l and s")
            s_portion = cv.getTrackbarPos("s", "l and s")
            # 调整饱和度和亮度
            hlsCopy[:, :, 1] = (1.0 + l_portion / float(MAX_VALUE)) * hlsCopy[:, :, 1]
            hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
            hlsCopy[:, :, 2] = (1.0 + s_portion / float(MAX_VALUE)) * hlsCopy[:, :, 2]
            hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1

            lsImg = cv.cvtColor(hlsCopy, cv.COLOR_HLS2BGR)
            cv.imshow("LandS_image", lsImg)
            ch = cv.waitKey(10)
            if ch == 27:
                break

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python python-scripy.py imageFile.")
