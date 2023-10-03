#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 08_QCheckBox_QRadioButton.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/02 20:17:35
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
from PySide6.QtWidgets import QCheckBox, QGroupBox, QRadioButton, QButtonGroup

import sys


class Widget(QWidget):
    def __init__(self, width=400, height=300):
        super().__init__()
        self.setWindowTitle("QCheckBox and QRadioButton")
        self.resize(width, height)

        # ======== Checkboxes: operating system ========
        os = QGroupBox("Choose operating system")
        
        windows = QCheckBox("Windows")
        windows.toggled.connect(self.windows_box_toggled)

        linux = QCheckBox("Linux")
        linux.toggled.connect(self.linux_box_toggled)

        mac = QCheckBox("Mac")
        mac.toggled.connect(self.mac_box_toggled)

        # the UI layout
        os_layout = QVBoxLayout()
        os_layout.addWidget(windows)
        os_layout.addWidget(linux)
        os_layout.addWidget(mac)
        os.setLayout(os_layout)

        # ======= Exclusive checkboxes: Drinks =======
        drinks = QGroupBox("Choose your drink")

        beer = QCheckBox("Beer")
        juice = QCheckBox("Juice")
        coffe = QCheckBox("Coffe")
        beer.setChecked(True)

        # Make the checkboxes exclusive
        exclusive_button_group = QButtonGroup(self) # The self parent is needed here.
        exclusive_button_group.addButton(beer)
        exclusive_button_group.addButton(juice)
        exclusive_button_group.addButton(coffe)
        exclusive_button_group.setExclusive(True)

        # the UI layout
        drink_layout = QVBoxLayout()
        drink_layout.addWidget(beer)
        drink_layout.addWidget(juice)
        drink_layout.addWidget(coffe)
        drinks.setLayout(drink_layout)

        # ======== Radio buttons: answers ========
        answers = QGroupBox("Choose Answer")
        answer_a = QRadioButton("A")
        answer_b = QRadioButton("B")
        answer_c = QRadioButton("C")
        answer_a.setChecked(True)

        # the UI layout
        answers_layout = QVBoxLayout()
        answers_layout.addWidget(answer_a)
        answers_layout.addWidget(answer_b)
        answers_layout.addWidget(answer_c)
        answers.setLayout(answers_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(os)
        h_layout.addWidget(drinks)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(answers)

        self.setLayout(v_layout)

    def windows_box_toggled(self,checked): 
        if(checked):
            print("Windows box checked")
        else:
            print("Windows box unchecked")

    def linux_box_toggled(self,checked): 
        if(checked):
            print("Linux box checked")
        else:
            print("Linux box unchecked")

    def mac_box_toggled(self,checked): 
        if(checked):
            print("Mac box checked")
        else:
            print("Mac box unchecked")


# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = Widget()

    window.show()
    app.exec()
