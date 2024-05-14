"""
TODO:
Clean up:
-remove all slot function references (the funcs should go to update_objects.py, the calls to them should be made by
tabs.py
- double check that the way things are defined makes sense (e.g. hard-coding things may not be ideal)
"""

from PyQt5.QtWidgets import *
import pyqtgraph as pg


class Display(QWidget):
    def __init__(self):
        super().__init__()

        # Data Objects, labels, buttons
        # Epoch
        self.epoch = 0

        # epoch label
        self.epoch_win = QLabel("Epoch: {} ".format(self.epoch), self)

        # current file label
        self.file_win = QLabel("", self)

        # current score label
        self.score_win = QLabel("", self)

        # create plot objects in window
        self.window_size = QComboBox(self)

        # Buttons
        self.load_data_button = QPushButton("Load Data", self)
        self.load_data_button.clicked.connect(self.Buttons.load_data)

        # EEG plot
        self.eeg_plot = pg.PlotWidget(self, title="EEG")
        self.line1 = pg.InfiniteLine(
            pos=80, movable=True, angle=0, pen=pg.mkPen(width=3, color="r")
        )
        self.line2 = pg.InfiniteLine(
            pos=-80, movable=True, angle=0, pen=pg.mkPen(width=3, color="r")
        )

        # EMG plot
        self.emg_plot = pg.PlotWidget(self, title="EMG")

        # Power Spectrum Line plot
        self.eeg_power_plot = pg.PlotWidget(self, title="Power Spectrum")

        # Power Spectrum Bar plot
        self.eeg_bar_plot = pg.PlotWidget(self, title="Relative Power")

        self.rel_delta = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.rel_theta = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.labels = {0: "delta", 1: "theta"}
        (self.eeg_bar_plot.getAxis("bottom")).setTicks([self.labels.items()])

        # Hypnogram
        self.hypnogram = pg.PlotWidget(self, title="Hypnogram")
        self.hyp_labels = {0: "Wake", 1: "Non REM", 2: "REM", 3: "Unscored"}
        (self.hypnogram.getAxis("left")).setTicks([self.hyp_labels.items()])
        self.hypno_line = pg.InfiniteLine(
            pos=self.epoch, movable=True, angle=90, pen=pg.mkPen(width=3, color="r")
        )
