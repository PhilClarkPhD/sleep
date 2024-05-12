from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import os
import windows


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        central_widget = QWidget()  # Create a central widget
        self.setCentralWidget(
            central_widget
        )  # Set it as the central widget of the main window

        layout = QVBoxLayout(central_widget)  # Create a layout for the central widget

        tabs = QTabWidget()
        tabs.addTab(windows.Tab().home(), "home")
        tabs.addTab(windows.Tab().model(), "Model")

        layout.addWidget(tabs)  # Add the tabs widget to the layout


def run():
    app = QApplication([])
    GUI = MainWindow()
    GUI.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
