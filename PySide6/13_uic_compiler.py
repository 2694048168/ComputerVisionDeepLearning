#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 13_uic_compiler.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/03 11:47:53
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

import sys

from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget

from ui_widget import Ui_Widget


class Widget(QWidget, Ui_Widget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("User data")
        self.submit_button.clicked.connect(self.do_something)
        
    def do_something(self):
        print(self.full_name_line_edit.text()," is a ",self.occupation_line_edit.text())


# --------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    window = Widget()
    window.show()

    app.exec()
