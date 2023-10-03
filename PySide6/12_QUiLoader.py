#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 12_QUiLoader.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/03 11:39:23
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

import sys

from PySide6 import QtWidgets, QtCore
from PySide6.QtUiTools import QUiLoader


# An object wrapping around our ui
class UserInterface(QtCore.QObject): 
    def __init__(self, loader):
        super().__init__()
        self.ui = loader.load("12_widget.ui", None)
        self.ui.setWindowTitle("User Data")
        self.ui.submit_button.clicked.connect(self.do_something)

    def show(self):
        self.ui.show()

    def do_something(self):
        print(self.ui.full_name_line_edit.text()," is a ",self.ui.occupation_line_edit.text())


# --------------------------
if __name__ == "__main__":
    # VERSION: the QUILoader to load the .ui file from QT Designer
    """
    # Set up a loader object
    loader = QUiLoader()

    app = QtWidgets.QApplication(sys.argv)

    # Load the ui - happens at run time!
    window = loader.load("12_widget.ui", None)

    def do_something() :
        print(window.full_name_line_edit.text(),"is a ", window.occupation_line_edit.text())

    # Changing the properties in the form
    window.setWindowTitle("User data")

    # Accessing widgets in the form
    window.submit_button.clicked.connect(do_something)
    window.show()

    app.exec()
    """

    # VERSION: the QUILoader to load the .ui file from QT Designer
    loader = QUiLoader()

    app = QtWidgets.QApplication(sys.argv)

    window = UserInterface(loader)
    window.show()

    app.exec()
