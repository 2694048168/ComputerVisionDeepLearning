#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 03_qwidget.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/02 15:43:35
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout

import sys


class CustomWidget(QWidget):
    def __init__(self, width=400, height=300):
        super().__init__()
        self.setWindowTitle("CustomWidget")
        self.resize(width, height)

        button1 = QPushButton("Button1")
        button1.clicked.connect(self.button1_clicked)
        button2 = QPushButton("Button2")
        button2.clicked.connect(self.button2_clicked)

        button_layout = QHBoxLayout()
        # button_layout = QVBoxLayout()
        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        self.setLayout(button_layout)

    def button1_clicked(self):
        print("button1 clicked")

    def button2_clicked(self):
        print("button2 clicked")


# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = CustomWidget()
    window.show()

    app.exec()
