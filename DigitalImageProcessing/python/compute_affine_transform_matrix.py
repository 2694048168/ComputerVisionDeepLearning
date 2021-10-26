#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 已知原始坐标和变换后的坐标，计算仿射变换矩阵
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-12
"""


import numpy as np
import cv2 as cv


def compute_affine_transform_matrix(src, dst):
    """利用 OpenCV 的 Python API 计算仿射变换矩阵。

    Args:
        src (float): 3 行 2 列的二维 ndarray，每一行代表一个原始坐标
        dst (float): 3 行 2 列的二维 ndarray，每一行代表一个变换后的坐标

    Returns:
        float: 返回仿射变换矩阵(3行3列)的前两行(2行3列)，
            最后一行为[0, 0, 1](齐次坐标表示法)
    """
    return cv.getAffineTransform(src=src, dst=dst)


def compute_affine_transform(scale, translation):
    """数字图像首先经过缩放变换，在进行平移变换。
    利用基本的仿射变换进行矩阵乘法求解出仿射变换矩阵。

    Args:
        scale (np.array): 基本的仿射变换——缩放
        translation (ndarray): 基本的仿射变换——平移
    
    Returns:
        float: 返回仿射变换矩阵(3行3列)
    """
    # 注意计算方向
    # return np.dot(scale, translation)
    return np.dot(translation, scale)


def compute_affine_transform_ratio(center, angle, scale):
    """等比例进行缩放并进行旋转。
    计算仿射变换矩阵，其本质还是计算各个基本的仿射矩阵进行相乘计算。

    Args:
        center ([type]): 变换中心的坐标
        angle ([type]): 等比例缩放的系数
        scale ([type]): 逆时针旋转的角度，单位为角度，而不是弧度；该值为负数，即为顺时针旋转。

    Returns:
        [type]: 通过旋转，平移，缩放等基本操作后的仿射变换矩阵(齐次坐标表示的前两行数值)
    """

    return cv.getRotationMatrix2D(center=center, angle=angle, scale=scale)


# --------------------------
if __name__ == "__main__":
    # 1. 方程方法
    src = np.array([[0, 0], [200, 0], [0, 200]], np.float32)
    dst = np.array([[0, 0], [100, 0], [0, 100]], np.float32)
    affine_transform_matrix = compute_affine_transform_matrix(src=src, dst=dst)
    print(f"The affine transform maxtrix type is\n {type(affine_transform_matrix)}")
    print(f"The affine transform maxtrix data type is\n {affine_transform_matrix.dtype}")
    print(f"The affine transform maxtrix is\n {affine_transform_matrix}")

    # 2. 矩阵方法
    scale = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    translation = np.array([[1, 0, 100], [0, 1, 200], [0, 0, 1]])
    affine_transform = compute_affine_transform(scale=scale, translation=translation)
    print(f"The affine transform maxtrix type is\n {type(affine_transform)}")
    print(f"The affine transform maxtrix data type is\n {affine_transform.dtype}")
    print(f"The affine transform maxtrix is\n {affine_transform}")

    # 3. 计算 (40, 50)为中心点进行逆时针旋转30度的仿射变换矩阵
    rotataion_matrix = compute_affine_transform_ratio(center=(40, 50), angle=30, scale=0.5)
    print(f"The affine transform maxtrix type is\n {type(rotataion_matrix)}")
    print(f"The affine transform maxtrix data type is\n {rotataion_matrix.dtype}")
    print(f"The affine transform maxtrix is\n {rotataion_matrix}")
