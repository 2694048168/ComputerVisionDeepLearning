#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 04_qmainwindow.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/02 15:53:49
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWidgets import QToolBar, QPushButton, QStatusBar
from PySide6.QtGui import QAction, QIcon

import sys


class MainWindow(QMainWindow):
    def __init__(self, app, width=800, height=600):
        super().__init__()
        self.app = app
        self.setWindowTitle("Custom MainWindow")
        self.resize(width, height)

        # =========== Menubar and menus ===========
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        open_action = file_menu.addAction("&Open")
        close_action = file_menu.addAction("&Close")
        quit_action = file_menu.addAction("&Quit")

        quit_action.triggered.connect(self.quit_app)

        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction("Copy")
        edit_menu.addAction("Cut")
        edit_menu.addAction("Paste")
        edit_menu.addAction("Undo")
        edit_menu.addAction("Redo")

        # A bunch of other menu options just for the fun of it
        menu_bar.addMenu("Window")
        menu_bar.addMenu("Setting")
        menu_bar.addMenu("Help")

        # =========== Working with toolbars ===========
        toolbar = QToolBar("My main toolbar")
        toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(toolbar)

        # Add the quit action to the toolbar
        toolbar.addAction(quit_action)

        action1 = QAction("Some Action", self)
        action1.setStatusTip("Status message for some action")
        action1.triggered.connect(self.toolbar_button_click)
        toolbar.addAction(action1)

        action2 = QAction(QIcon("images/start.png"), "Some other action", self)
        action2.setStatusTip("Status message for some other action")
        action2.triggered.connect(self.toolbar_button_click)
        #action2.setCheckable(True)
        toolbar.addAction(action2)

        toolbar.addSeparator()
        toolbar.addWidget(QPushButton("Click here"))

        # =========== Working with status bars ===========
        self.setStatusBar(QStatusBar(self))

        button1 = QPushButton("BUTTON1")
        button1.clicked.connect(self.button1_clicked)
        self.setCentralWidget(button1)

    def quit_app(self):
        self.app.quit()   

    def button1_clicked(self):
        print("Clicked on the button")

    def toolbar_button_click(self):
        self.statusBar().showMessage("Message from my app", 1000)


# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow(app)
    window.show()

    app.exec()
