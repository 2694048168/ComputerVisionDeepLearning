from PySide6.QtWidgets import QMainWindow, QPushButton

class ButtonHolder(QMainWindow):
    def __init__ (self, width=800, height=600):
        super().__init__()
        self.setWindowTitle("Button Holder")
        self.resize(width, height)
        button = QPushButton("Press Me!")

        # Set up the button as our central widget
        self.setCentralWidget(button)
        