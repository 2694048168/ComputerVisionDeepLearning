#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 06_components.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/02 16:48:13
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout
from PySide6.QtWidgets import QPushButton, QLineEdit, QLabel, QTextEdit
from PySide6.QtGui import QPixmap

import sys


# components of QWidgets: QPushButton
class WidgetQPushButton(QWidget):
    def __init__(self, width=400, height=300):
        super().__init__()
        self.setWindowTitle("Custom MainWindow")
        self.resize(width, height)

        button = QPushButton("Click")
        button.clicked.connect(self.button_clicked)
        button.pressed.connect(self.button_pressed)
        button.released.connect(self.button_released)

        layout = QVBoxLayout()
        layout.addWidget(button)

        self.setLayout(layout)

    def button_clicked(self):
        print("Clicked")

    def button_pressed(self): 
        print("Pressed")

    def button_released(self):
        print("Released")


# components of QWidgets: QLineEdit & QLabel
class WidgetQLineEdit(QWidget):
    def __init__(self, width=400, height=100):
        super().__init__()
        self.setWindowTitle("QLabel and QLineEdit")
        self.resize(width, height)

        # A set of signals we can connect to
        label = QLabel("Fullname: ")
        self.line_edit = QLineEdit()
        self.line_edit.textChanged.connect(self.text_changed)
        self.line_edit.cursorPositionChanged.connect(self.cursor_position_changed)
        self.line_edit.editingFinished.connect(self.editing_finished)
        self.line_edit.returnPressed.connect(self.return_pressed)
        self.line_edit.selectionChanged.connect(self.selection_changed)
        self.line_edit.textEdited.connect(self.text_edited)

        button = QPushButton("Grab data")
        button.clicked.connect(self.button_clicked)
        self.text_holder_label = QLabel("Wei Li GitHub 2694048168")

        h_layout = QHBoxLayout()
        h_layout.addWidget(label)
        h_layout.addWidget(self.line_edit)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(button)
        v_layout.addWidget(self.text_holder_label)

        self.setLayout(v_layout)

    # Slots
    def button_clicked(self):
        print(f"Fullname: {self.line_edit.text()}")
        self.text_holder_label.setText(self.line_edit.text())

    def text_changed(self):
        print(f"Text  changed to: {self.line_edit.text()}")
        self.text_holder_label.setText(self.line_edit.text())

    def cursor_position_changed(self,old,new):
        print(f"cursor old position: {old}, -new position: {new}")

    def editing_finished(self) : 
        print("Editing finished")

    def return_pressed(self):
        print("Return pressed")

    def selection_changed(self):
        print(f"Selection Changed: {self.line_edit.selectedText()}")

    def text_edited(self,new_text) : 
        print(f"Text edited. New text: {new_text}")

# components of QWidgets: QTextEdit
class WidgetQTextEdit(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QTextEdit Demo")

        self.text_edit = QTextEdit()
        # self.text_edit.textChanged.connect(self.text_changed)

        # Buttons
        current_text_button = QPushButton("Current Text")
        current_text_button.clicked.connect(self.current_text_button_clicked)

        copy_button = QPushButton("Copy")
        # Connect directly to QTextEdit slot
        copy_button.clicked.connect(self.text_edit.copy)

        cut_button = QPushButton("Cut")
        cut_button.clicked.connect(self.text_edit.cut)

        paste_button = QPushButton("Paste")
        # Go through a custom slot
        paste_button.clicked.connect(self.paste) 

        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self.text_edit.undo)

        redo_button = QPushButton("Redo")
        redo_button.clicked.connect(self.text_edit.redo)

        set_plain_text_button = QPushButton("Set Plain Text")
        set_plain_text_button.clicked.connect(self.set_plain_text)

        set_html_button = QPushButton("Set html")
        set_html_button.clicked.connect(self.set_html)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.text_edit.clear)

        # ========== layout for UI ==========
        h_layout = QHBoxLayout()
        h_layout.addWidget(current_text_button)
        h_layout.addWidget(copy_button)
        h_layout.addWidget(cut_button)
        h_layout.addWidget(paste_button)
        h_layout.addWidget(undo_button)
        h_layout.addWidget(redo_button)
        h_layout.addWidget(set_plain_text_button)
        h_layout.addWidget(set_html_button)
        h_layout.addWidget(clear_button)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.text_edit)

        self.setLayout(v_layout)

    def current_text_button_clicked(self):
        print(self.text_edit.toPlainText())

    def paste(self):
        self.text_edit.paste()

    def set_plain_text(self) : 
        self.text_edit.setPlainText("Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?")

    def set_html(self):
         self.text_edit.setHtml("<h1>Kigali Districts</h1><p>The city of Kigali has three districts : </br> <ul> <li>Gasabo</li> <li>Nyarugenge</li><li>Kicukiro</li></ul></p>")


# components of QWidgets: QLabel and images
class WidgetImage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QLabel Image Demo")

        image_label = QLabel()
        image_label.setPixmap(QPixmap("images/minions.png"))

        layout = QVBoxLayout()
        layout.addWidget(image_label)

        self.setLayout(layout)


# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # ======== components of QWidgets: QPushButton ========
    # window = WidgetQPushButton()

    # ======== components of QWidgets: QLineEdit & QLabel ========
    # window = WidgetQLineEdit()

    # ======== components of QWidgets: QTextEdit ========
    # window = WidgetQTextEdit()

    # ======== components of QWidgets: QLabel and images ========
    window = WidgetImage()

    window.show()
    app.exec()
