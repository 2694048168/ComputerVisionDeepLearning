#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: main.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/03 13:15:11
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

import sys
from PySide6.QtWidgets import QApplication

from mainwindow import MainWindow


# -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    w = MainWindow(app)
    w.show()

    app.exec()
