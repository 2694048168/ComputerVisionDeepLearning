#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 02_slot_signal.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/02 15:06:00
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QSlider
from PySide6.QtWidgets import QPushButton, QMainWindow

# VERSION1 : Just responding to the button click : syntax
# The slot : responds when something happens
def button_clicked():
    print("You clicked the button, didn't you!")


# VERSION2 : Signal sending values, capture values in Slots
# The slot : responds when something happens
def button_clicked(data):
    print(f"You clicked the button, didn't you! checked : {data}")


# VERSION3 : Capture value from a slider
# The slot : responds when something happens
def respond_to_slider(data):
    print("slider moved to : ", data)


# --------------------------
if __name__ == "__main__":
    app = QApplication()

    window = QMainWindow()
    window.setWindowTitle("Slot and Signal")
    window.resize(600, 400)

    button = QPushButton("Press Me")
    window.setCentralWidget(button)

    # Makes the button checkable. It's unchecked by default.
    # Further clicks toggle between checked and unchecked states
    button.setCheckable(True) 

    #clicked is a signal of QPushButton. 
    # It's emitted when you click on the button
    #You can wire a slot to the signal using the syntax below : 
    button.clicked.connect(button_clicked)

    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(100)
    slider.setValue(35)
    slider.resize(300, 200)

    # You just do the connection. 
    # The Qt system takes care of passing the data from the signal to the slot.
    slider.valueChanged.connect(respond_to_slider)
    slider.show()

    window.show()
    app.exec()
