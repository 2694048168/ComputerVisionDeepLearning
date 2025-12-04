#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: GrabCutGMM.py
@Python Version: 3.12.8
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2025-11-05
@copyright Copyright (c) 2025 Wei Li
@Description: 基于OpenCV GrabCut 交互式图像分割（树叶分割）算法实现
@Doc:
"""

import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSlider,
    QMessageBox,
    QGroupBox,
    QCheckBox,
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor


class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setText("请加载一张图像")
        self.setStyleSheet("border: 1px solid black;")

        # 启用鼠标跟踪
        self.setMouseTracking(True)

        # 绘图相关变量
        self.drawing = False
        self.drawing_mode = "rect"  # "rect" 或 "mask"
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.rect = None
        self.mask_points = []
        self.mask_type = "foreground"  # "foreground" 或 "background"

        # 图像相关变量
        self.original_image = None
        self.display_image = None
        self.result_image = None
        self.mask = None

        # 用于调试
        self.debug_info = ""

    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            return False

        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.display_image = self.original_image.copy()
        self.result_image = None
        self.mask = None
        self.rect = None
        self.mask_points = []

        self.update_display()
        return True

    def update_display(self):
        if self.display_image is None:
            return

        height, width, channel = self.display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            self.display_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        self.setPixmap(QPixmap.fromImage(q_image))

    def mousePressEvent(self, event):
        if self.original_image is None:
            return

        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()

            if self.drawing_mode == "mask":
                # 将点击点转换为图像坐标
                img_x, img_y = self.convert_to_image_coords(event.pos())

                if img_x is not None and img_y is not None:
                    self.mask_points.append((img_x, img_y, self.mask_type))

            self.update_display_with_drawing()

    def mouseMoveEvent(self, event):
        if self.drawing and self.original_image is not None:
            self.end_point = event.pos()
            self.update_display_with_drawing()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False

            if self.drawing_mode == "rect":
                # 将矩形坐标转换为图像坐标
                start_x, start_y = self.convert_to_image_coords(self.start_point)
                end_x, end_y = self.convert_to_image_coords(self.end_point)

                if start_x is not None and end_x is not None:
                    # 确保矩形坐标有效
                    x1 = min(start_x, end_x)
                    y1 = min(start_y, end_y)
                    x2 = max(start_x, end_x)
                    y2 = max(start_y, end_y)

                    # 确保矩形在图像范围内
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(self.original_image.shape[1] - 1, x2)
                    y2 = min(self.original_image.shape[0] - 1, y2)

                    self.rect = (x1, y1, x2 - x1, y2 - y1)

            self.update_display_with_drawing()

    def convert_to_image_coords(self, point):
        """将界面坐标转换为图像坐标"""
        pixmap = self.pixmap()
        if not pixmap:
            return None, None

        # 获取QLabel的尺寸
        label_width = self.width()
        label_height = self.height()

        # 获取图像的尺寸
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # 计算缩放比例
        scale_x = pixmap_width / self.original_image.shape[1]
        scale_y = pixmap_height / self.original_image.shape[0]

        # 计算图像在QLabel中的偏移量（居中显示）
        offset_x = (label_width - pixmap_width) / 2
        offset_y = (label_height - pixmap_height) / 2

        # 转换为图像坐标
        img_x = point.x() - offset_x
        img_y = point.y() - offset_y

        # 检查是否在图像范围内
        if 0 <= img_x < pixmap_width and 0 <= img_y < pixmap_height:
            # 转换为原始图像坐标
            orig_x = int(img_x / scale_x)
            orig_y = int(img_y / scale_y)

            return orig_x, orig_y

        return None, None

    def update_display_with_drawing(self):
        if self.original_image is None:
            return

        # 创建显示图像的副本
        self.display_image = self.original_image.copy()

        # 如果有分割结果，叠加显示
        if self.result_image is not None:
            # 创建一个掩码，只显示分割区域
            mask = self.mask.copy()
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = mask.astype(bool)

            # 将分割结果叠加到原图上
            self.display_image[mask] = (
                self.display_image[mask] * 0.5 + self.result_image[mask] * 0.5
            )

        # 绘制矩形
        if self.drawing_mode == "rect" and self.drawing:
            # 转换坐标到图像坐标系
            start_img_x, start_img_y = self.convert_to_image_coords(self.start_point)
            end_img_x, end_img_y = self.convert_to_image_coords(self.end_point)

            if start_img_x is not None and end_img_x is not None:
                # 确保坐标有效
                x1 = min(start_img_x, end_img_x)
                y1 = min(start_img_y, end_img_y)
                x2 = max(start_img_x, end_img_x)
                y2 = max(start_img_y, end_img_y)

                # 绘制矩形
                cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 绘制已确定的矩形
        elif self.rect is not None and self.drawing_mode == "rect":
            x, y, w, h = self.rect
            cv2.rectangle(self.display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 绘制掩码点
        for point in self.mask_points:
            x, y, mask_type = point
            color = (0, 255, 0) if mask_type == "foreground" else (0, 0, 255)
            cv2.circle(self.display_image, (x, y), 5, color, -1)

        self.update_display()

    def apply_grabcut(self):
        if self.original_image is None:
            return False

        # 转换为BGR格式供OpenCV使用
        img = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)

        # 初始化掩码
        if self.mask is None:
            self.mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 设置GrabCut参数
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # 如果有矩形，使用矩形初始化
        if self.rect is not None:
            x, y, w, h = self.rect
            self.mask[:] = cv2.GC_PR_BGD  # 可能的背景

            # 矩形内部设置为可能的前景
            self.mask[y : y + h, x : x + w] = cv2.GC_PR_FGD

            # 执行GrabCut
            cv2.grabCut(
                img,
                self.mask,
                self.rect,
                bgd_model,
                fgd_model,
                5,
                cv2.GC_INIT_WITH_RECT,
            )
        else:
            # 如果没有矩形，使用掩码初始化
            cv2.grabCut(
                img, self.mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK
            )

        # 应用掩码点
        for point in self.mask_points:
            x, y, mask_type = point
            if mask_type == "foreground":
                self.mask[y, x] = cv2.GC_FGD  # 确定的前景
            else:
                self.mask[y, x] = cv2.GC_BGD  # 确定的背景

        # 如果有掩码点，再次执行GrabCut
        if self.mask_points:
            cv2.grabCut(
                img, self.mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK
            )

        # 创建结果掩码
        result_mask = np.where(
            (self.mask == cv2.GC_FGD) | (self.mask == cv2.GC_PR_FGD), 255, 0
        ).astype("uint8")

        # 应用掩码到原图
        self.result_image = cv2.bitwise_and(img, img, mask=result_mask)
        self.result_image = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)

        # 更新显示
        self.update_display_with_drawing()

        return True

    def clear_drawing(self):
        self.rect = None
        self.mask_points = []
        self.mask = None
        self.result_image = None
        self.update_display_with_drawing()

    def set_drawing_mode(self, mode):
        self.drawing_mode = mode
        print(f"切换到 {mode} 模式")  # 调试信息

    def set_mask_type(self, mask_type):
        self.mask_type = mask_type
        print(f"切换到 {mask_type} 笔")  # 调试信息


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交互式树叶分割 - GrabCut算法")
        self.setGeometry(100, 100, 1000, 700)

        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # 左侧图像显示区域
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer, 4)

        # 右侧控制面板
        control_panel = QVBoxLayout()
        layout.addLayout(control_panel, 1)

        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)

        self.load_button = QPushButton("加载图像")
        self.load_button.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_button)

        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_result)
        file_layout.addWidget(self.save_button)

        control_panel.addWidget(file_group)

        # 分割操作组
        segment_group = QGroupBox("分割操作")
        segment_layout = QVBoxLayout(segment_group)

        self.rect_mode_button = QPushButton("矩形模式")
        self.rect_mode_button.clicked.connect(
            lambda: self.image_viewer.set_drawing_mode("rect")
        )
        segment_layout.addWidget(self.rect_mode_button)

        self.mask_mode_button = QPushButton("掩码模式")
        self.mask_mode_button.clicked.connect(
            lambda: self.image_viewer.set_drawing_mode("mask")
        )
        segment_layout.addWidget(self.mask_mode_button)

        self.foreground_button = QPushButton("前景笔")
        self.foreground_button.clicked.connect(
            lambda: self.image_viewer.set_mask_type("foreground")
        )
        segment_layout.addWidget(self.foreground_button)

        self.background_button = QPushButton("背景笔")
        self.background_button.clicked.connect(
            lambda: self.image_viewer.set_mask_type("background")
        )
        segment_layout.addWidget(self.background_button)

        self.segment_button = QPushButton("执行分割")
        self.segment_button.clicked.connect(self.apply_segmentation)
        segment_layout.addWidget(self.segment_button)

        self.clear_button = QPushButton("清除标记")
        self.clear_button.clicked.connect(self.image_viewer.clear_drawing)
        segment_layout.addWidget(self.clear_button)

        control_panel.addWidget(segment_group)

        # 显示选项组
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout(display_group)

        self.show_original_check = QCheckBox("显示原图")
        self.show_original_check.setChecked(True)
        self.show_original_check.stateChanged.connect(self.toggle_display)
        display_layout.addWidget(self.show_original_check)

        self.show_result_check = QCheckBox("显示分割结果")
        self.show_result_check.setChecked(True)
        self.show_result_check.stateChanged.connect(self.toggle_display)
        display_layout.addWidget(self.show_result_check)

        control_panel.addWidget(display_group)

        control_panel.addStretch()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            if self.image_viewer.load_image(file_path):
                self.statusBar().showMessage(f"已加载图像: {file_path}")
                print(f"已加载图像: {file_path}")  # 调试信息
            else:
                QMessageBox.warning(self, "错误", "无法加载图像文件")

    def save_result(self):
        if self.image_viewer.result_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的分割结果")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存分割结果", "", "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg)"
        )

        if file_path:
            # 转换为BGR格式保存
            result_bgr = cv2.cvtColor(self.image_viewer.result_image, cv2.COLOR_RGB2BGR)
            if cv2.imwrite(file_path, result_bgr):
                self.statusBar().showMessage(f"已保存结果: {file_path}")
            else:
                QMessageBox.warning(self, "错误", "保存图像失败")

    def apply_segmentation(self):
        if self.image_viewer.original_image is None:
            QMessageBox.warning(self, "警告", "请先加载一张图像")
            return

        if self.image_viewer.rect is None and not self.image_viewer.mask_points:
            QMessageBox.warning(self, "警告", "请先绘制矩形区域或添加掩码点")
            return

        if self.image_viewer.apply_grabcut():
            self.statusBar().showMessage("分割完成")
        else:
            self.statusBar().showMessage("分割失败")

    def toggle_display(self):
        # 更新显示逻辑
        self.image_viewer.update_display_with_drawing()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
