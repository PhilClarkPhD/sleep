from PyQt5.QtWidgets import *
import sys
from tabs import Tabs
import pyqtgraph as pg
from update_objects import Funcs

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setGeometry(50, 50, 1125, 525)
        self.setWindowTitle("Mora Sleep Analysis")
        self.setStyleSheet("font-size: 20px;")

        self.Funcs = Funcs()
        self.Tabs = Tabs()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        tabs.addTab(self.Tabs.home_tab(), "Home")
        tabs.addTab(self.Tabs.model_tab(), "Model")

        layout.addWidget(tabs)

    def keyPressEvent(self, event):  # Needed to ensure key press functionality
        self.Tabs.Funcs.keyPressEvent(event)


def run():
    app = QApplication([])
    GUI = MainWindow()
    GUI.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
