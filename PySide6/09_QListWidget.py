#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 09_QListWidget.py
@Python Version: 3.11.4
@Platform: PySide6 6.5.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/10/02 22:53:47
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: 
@Description: 
'''

from PySide6.QtWidgets import QApplication, QWidget, QAbstractItemView
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
from PySide6.QtWidgets import QListWidget, QPushButton

import sys

class Widget(QWidget):
    def __init__(self, width=400, height=300):
        super().__init__()
        self.setWindowTitle("QListWidget Example")
        self.resize(width, height)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_widget.addItem("One")
        self.list_widget.addItems(["Two","Three"])

        self.list_widget.currentItemChanged.connect(self.current_item_changed)
        self.list_widget.currentTextChanged.connect(self.current_text_changed)

        button_add_item = QPushButton("Add Item")
        button_add_item.clicked.connect(self.add_item)

        button_delete_item = QPushButton("Delete Item")
        button_delete_item.clicked.connect(self.delete_item)

        button_item_count = QPushButton("Item Count")
        button_item_count.clicked.connect(self.item_count)

        button_selected_items = QPushButton("Selected Items")
        button_selected_items.clicked.connect(self.selected_items)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.list_widget)

        v_layout.addWidget(button_add_item)
        v_layout.addWidget(button_delete_item)
        v_layout.addWidget(button_item_count)
        v_layout.addWidget(button_selected_items)

        self.setLayout(v_layout)

    def current_item_changed(self, item):
        print("Current item : ",item.text())

    def current_text_changed(self,text):
        print("Current text changed : ",text)

    def add_item(self):
        self.list_widget.addItem("New Item")

    def item_count(self):
        print("Item count : ",self.list_widget.count())

    def delete_item(self):
        self.list_widget.takeItem(self.list_widget.currentRow())

    def selected_items(self):
        list = self.list_widget.selectedItems()
        for i in list : 
            print(i.text())


# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = Widget()

    window.show()
    app.exec()
