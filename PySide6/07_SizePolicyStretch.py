#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 07_SizePolicyStretch.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/02 17:40:21
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QSizePolicy, QGridLayout

import sys


# Size Policy and Stretch
class Widget(QWidget):
    def __init__(self, width=200, height=100):
        super().__init__()
        self.setWindowTitle("Size policies and stretches")
        self.resize(width, height)

        # Size policy in QT UserInterface:
        # how the widgets behaves if container space is expanded or shrunk.
        label = QLabel("Some text : ")
        line_edit = QLineEdit()

        line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # line_edit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        h_layout_1 = QHBoxLayout()
        h_layout_1.addWidget(label)
        h_layout_1.addWidget(line_edit)

        button_1 = QPushButton("One")
        button_2 = QPushButton("Two")
        button_3 = QPushButton("Three")

        # stretch in QT UserInterface:
        # how much of the available space (in the layout) is occupied by each widget.
        # You specify the stretch when you add things to the layout:
        # button_1 takes up 2 units,button_2 and button_3 each take up 1 unit
        # ======== the QT UserInterface layout ========
        h_layout_2 = QHBoxLayout()
        h_layout_2.addWidget(button_1, 2)
        h_layout_2.addWidget(button_2, 1)
        h_layout_2.addWidget(button_3, 1)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout_1)
        v_layout.addLayout(h_layout_2)
        
        self.setLayout(v_layout)


# QtWidgets QGridLayout module: the layout for QT User Interface
class WidgetGridLayout(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QGridLayout Demo")

        label = QLabel("Some text : ")
        line_edit = QLineEdit()
        line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        button_1 = QPushButton("One")
        button_2 = QPushButton("Two")
        button_3 = QPushButton("Three")
        button_3.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        button_4 = QPushButton("Four")
        button_5 = QPushButton("Five")
        button_6 = QPushButton("Six")
        button_7 = QPushButton("Seven")

        h_layout_1 = QHBoxLayout()
        h_layout_1.addWidget(label)
        h_layout_1.addWidget(line_edit)

        # he layout for QT User Interface
        grid_layout = QGridLayout()
        # position(0, 0) in the Grid layout
        grid_layout.addWidget(button_1, 0, 0)
        # position(0, 1) in the Grid layout, and Take up 1 row and 2 columns
        grid_layout.addWidget(button_2, 0, 1, 1, 2)
        # position(1, 0) in the Grid layout, and Take up 2 rows and 1 column
        grid_layout.addWidget(button_3, 1, 0, 2, 1)
        grid_layout.addWidget(button_4, 1, 1)
        grid_layout.addWidget(button_5, 1, 2)
        grid_layout.addWidget(button_6, 2, 1)
        grid_layout.addWidget(button_7, 2, 2)

        global_layout = QVBoxLayout()
        global_layout.addLayout(h_layout_1)
        global_layout.addLayout(grid_layout)

        self.setLayout(global_layout)


# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # window = Widget()
    window = WidgetGridLayout()

    window.show()
    app.exec()
