#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 10_QTabWidget.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/02 23:06:47
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
from PySide6.QtWidgets import QTabWidget, QPushButton, QLabel, QLineEdit

import sys

class Widget(QWidget):
    def __init__(self, width=400, height=300):
        super().__init__()
        self.setWindowTitle("QTabWidget Example")
        self.resize(width, height)

        tab_widget = QTabWidget(self)

        # Information
        widget_form = QWidget()
        label_full_name = QLabel("Full name :")
        line_edit_full_name = QLineEdit()
        form_layout = QHBoxLayout()
        form_layout.addWidget(label_full_name)
        form_layout.addWidget(line_edit_full_name)
        widget_form.setLayout(form_layout)

        # Buttons
        widget_buttons = QWidget()
        button_1 = QPushButton("One")
        button_1.clicked.connect(self.button_1_clicked)
        button_2 = QPushButton("Two")
        button_3 = QPushButton("Three")
        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(button_1)
        buttons_layout.addWidget(button_2)
        buttons_layout.addWidget(button_3)
        widget_buttons.setLayout(buttons_layout)

        # Add tabs to widget
        tab_widget.addTab(widget_form,"Information")
        tab_widget.addTab(widget_buttons,"Button")

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)

    def button_1_clicked(self):
        print("Button clicked")


# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = Widget()

    window.show()
    app.exec()
