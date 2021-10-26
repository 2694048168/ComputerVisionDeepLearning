#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 轮廓
        查找、绘制轮廓
        外包、拟合轮廓
        轮廓的周长和面积
        点和轮廓的位置关系
        轮廓的凸包缺陷
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-19
"""

import sys
import numpy as np
import cv2 as cv


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage", image)

        # 边缘二值图
        img = cv.GaussianBlur(image, (3, 3), 0.5)
        binaryImg = cv.Canny(img, 50, 200)
        cv.imshow("BinaryCanny", binaryImg)

        # ------------ step 1, 查找、绘制轮廓 ------------
        # 边缘轮廓
        contours, h = cv.findContours(binaryImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        print(f"The data type of contour: {type(contours)}")
        print(f"The data shape of contour: {contours[0].shape}")
        contoursImg = []
        # 绘制轮廓
        for i in range(len(contours)):
            temp = np.zeros(binaryImg.shape, np.uint8)
            contoursImg.append(temp)
            # 在第 i 个黑色画布上, 绘制 i 个轮廓
            cv.drawContours(contoursImg[i], contours, i, 255, 2)
            cv.imshow(f"contour-{str(i)}", contoursImg[i])

        # # ------------ step 1, 查找、绘制轮廓 ------------
        # cv.findContours()
        # cv.drawContours()
        # # ------------ step 2, 外包、拟合轮廓 ------------
        # cv.approxPolyDP()
        # cv.boundingRect()
        # # ------------ step 3, 轮廓的周长和面积 ------------
        # cv.arcLength()
        # cv.contourArea()
        # # ------------ step 4, 点和轮廓的位置关系 ------------
        # cv.pointPolygonTest()
        # # ------------ step 5, 轮廓的凸包缺陷 ------------
        # cv.convexityDefects()

        cv.waitKey()
        cv.destroyAllWindows()

    else:
        print("Usage: python hough_transform.py imageFile")
