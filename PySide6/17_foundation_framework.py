#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 00_foundation_framework.py
@Python Version: 3.12.1
@Platform: PyTorch 2.0.0 + cu117
@Author: Wei Li (Ithaca)
@Date: 2024-07-10.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
"""

from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QMainWindow
import sys


class FoundationWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = FoundationWindow()
    window.show()

    app.exec()
