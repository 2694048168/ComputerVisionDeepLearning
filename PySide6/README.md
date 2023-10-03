## [Qt for Python: PySide6](https://wiki.qt.io/Qt_for_Python)

### **Features**
- [x] Python & VSCode
- [x] QWidgets & QApplication & QWidget & QPushButton
- [x] Slot and Signal mechanism in QT
- [x] The QtWidgets components: QWidget & QMainWindow & QMessageBox
- [x] The QtWidgets components: QPushButton & QLabel(Image) & QLineEdit & QTextEdit
- [x] QtWidgets QSizePolicy module: Size Policy and Stretch in QT User Interface
- [x] QtWidgets QGridLayout module: the layout for QT User Interface
- [x] QtWidgets QCheckBox and QRadioButton modules
- [x] The QtWidgets components: QListWidget & QTabWidget & QComboBox
- [x] Qt Designer: build layouts by just dragging and dropping
- [x] [Designer(.ui file)](https://doc.qt.io/qtforpython-6/tutorials/basictutorial/uifiles.html) --> PySide6.QtUiTools.QUiLoader --> uic compiler(.py file)
- [x] [Using Resources(.qrc file)](https://doc.qt.io/qtforpython-6/tutorials/basictutorial/qrcfiles.html): Manual or Automatic via Designer
- [x] Qt for Python: Events & Graphics View Framework & Networks & Databases & Threads


### quick start

```shell
# better advice that installing Miniconda for python 3.11.4

# install PySide6
pip install pyside6

python 00_main_widget.py

# using "uic" compiler to generate .py file or .h file
pyside6-uic 12_widget.ui -o 13_ui_widget.py

# using 'rcc' compiler to generate .py file
pyside6-rcc icons.qrc -o rc_icons.py

```


> [Reference Code](https://github.com/rutura/Qt-For-Python-PySide6-GUI-For-Beginners-The-Fundamentals-)