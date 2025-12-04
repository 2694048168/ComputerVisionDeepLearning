#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: ComputerMonitorApp.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-08-11.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V1.0
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: 
"""

import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
