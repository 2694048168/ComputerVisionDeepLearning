#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 点集的最小外包：圆形、直立矩阵、旋转矩阵、三角形、凸多边形
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-18
"""

import numpy as np
import cv2 as cv
            

# --------------------------
if __name__ == "__main__":
    print("-----------------------------------------------------")
    # -------- 点集的最小外包 --------
    # step 1, 最小外包旋转矩形
    # way 1 of point set store.
    pointSet = np.array([[1, 1], [5, 1], [1, 10], [5, 10], [2, 5]], np.int32)
    # way 2 of point set store.
    # pointSet = np.array([[[1, 1]], [[5, 1]], [[1, 10]], [[5, 10]], [[2, 5]]], np.int32)

    # compute the minimum outsourcing rotate rectangle of point set
    rotateRectangle = cv.minAreaRect(pointSet)
    # show infomation of rotate rectangle: [旋转矩阵的中心坐标，尺寸大小，旋转角度]
    print(f"The information of rotate rectangle: \n{rotateRectangle}")

    # 旋转矩形是通过中心点坐标、尺寸和旋转角度三个方面来定义的，
    # 通过这三个属性值就可以计算出旋转矩形的 4 个顶点，这样虽然简单，但是写起来比较复杂。
    # rotate rectangel
    vertices = cv.boxPoints(((200, 200), (90, 150), -60.0))
    print(f"The data type of rotate rectangle: {vertices.dtype}")
    print(f"The four points of rotate rectangle:\n {vertices}")
    # plot rotate rectangle with this four points
    img = np.zeros((400, 400), np.uint8)
    for idx in range(4):
        point_1 = vertices[idx, :]
        j = (idx+1) % 4
        point_2 = vertices[j, :]
        # plot line
        cv.line(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])), 255, 2)
    cv.imshow("RotateRectangle", img)

    print("-----------------------------------------------------")
    # step 2, 最小外包圆形
    pointSetCircle = np.array([[1, 1], [5, 1], [1, 10], [5, 10], [2, 5]], np.int32)
    circle = cv.minEnclosingCircle(pointSetCircle)
    print(f"The infomation of this minimum outsouring circle: \n{circle}")
    
    print("-----------------------------------------------------")
    # step 3, 最小外包直立矩形
    pointSetRect = np.array([[[1, 1]], [[5, 10]], [[5, 1]], [[1, 10]], [[2, 5]]], np.float32)
    rightRect = cv.boundingRect(pointSetRect)
    # 返回结果是一个由 4 个元素组成的元组，
    # 前两个元素是直立矩形的一个顶点坐标，
    # 后两个元素是它的对角坐标
    print(f"The infomation of smallest outsourcing upright rectangle: \n{rightRect}")

    print("-----------------------------------------------------")
    # step 4, 最小凸包
    # 黑色画布 400X400
    convexImage = np.zeros((400, 400), np.uint8)
    # 随机生成 横纵 坐标均在 100-300 之间的坐标点
    num_randomPoint = 80
    pointSetConvex = np.random.randint(100, 300, (num_randomPoint, 2), np.int32)
    # visualization of this points
    for i in range(num_randomPoint):
        cv.circle(convexImage, (pointSetConvex[i, 0], pointSetConvex[i, 1]), 2, 255, 2)
    
    # compute the convex outsourcing of this poin set
    convex_hull = cv.convexHull(pointSetConvex)
    # 打印最外侧的点(连接起来即为凸多边形)
    print(f"The data type: {type(convex_hull)}")
    print(f"The data shape: {convex_hull.shape}")

    # 一次连接凸包里面的点
    k = convex_hull.shape[0]
    for i in range(k-1):
        cv.line(convexImage, (convex_hull[i, 0, 0], convex_hull[i, 0, 1]), (convex_hull[i+1, 0, 0], convex_hull[i+1, 0, 1]), 255, 2)
    # 首尾坐标相连
    cv.line(convexImage, (convex_hull[k-1, 0, 0], convex_hull[k-1, 0, 1]), (convex_hull[0, 0, 0], convex_hull[0, 0, 1]), 255, 2)
    cv.imshow("ConvexImage", convexImage)
    
    print("-----------------------------------------------------")
    # step 5, 最小外包三角形
    pointSetTriangle = np.array([[[1, 1]], [[5, 10]], [[5, 1]], [[1, 10]], [[2, 5]]], np.float32)
    area, triangle = cv.minEnclosingTriangle(pointSetTriangle)
    print(f"The area of triangle: {area}")
    print(f"The three point of triangle:\n {triangle}")
    print("-----------------------------------------------------")

    cv.waitKey()
    cv.destroyAllWindows()
