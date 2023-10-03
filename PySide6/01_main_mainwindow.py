#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 01_main_mainwindow.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/01 21:21:56
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

# VERSION1 : Setting everything up in the global scope
"""
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
import sys

# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle("First MainWindow")
    window.resize(800, 600)

    button = QPushButton()
    button.setText("Press Me")

    window.setCentralWidget(button)

    window.show() 
    app.exec()
"""

# VERSION2 : Setting up a separate class
"""
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

class ButtonHolder(QMainWindow):
    def __init__ (self, width=800, height=600):
        super().__init__()
        self.setWindowTitle("Button Holder")
        self.resize(width, height)
        button = QPushButton("Press Me!")

        # Set up the button as our central widget
        self.setCentralWidget(button)

# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # window = ButtonHolder(400, 300)
    window = ButtonHolder()
    window.show() 

    app.exec()
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

# from button_holder import ButtonHolder
class ButtonHolder(QMainWindow):
    def __init__ (self, width=800, height=600):
        super().__init__()
        self.setWindowTitle("Button Holder")
        self.resize(width, height)
        button = QPushButton("Press Me!")

        # Set up the button as our central widget
        self.setCentralWidget(button)
        

# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ButtonHolder()
    window.show()

    app.exec()
