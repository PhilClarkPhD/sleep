from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
from PyQt5.QtCore import *
import numpy as np


class Display(QWidget):
    def __init__(self):
        super(QWidget, self).__init__()

        # epoch
        self.epoch = 0

        # epoch label
        self.epoch_win = QLabel("Epoch: {} ".format(self.epoch), self)

        # current file label
        self.file_win = QLabel("", self)

        # current score label
        self.score_win = QLabel("", self)

        # create plot objects in window
        self.window_size = QComboBox(self)

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

    def update_plots(self):
        x_labels = self.calculate_eeg_axes()

        # eeg plot
        self.eeg_plot.plot(
            x=self.x_axis,
            y=self.df["eeg"][self.x_axis[0] : self.x_axis[-1] + 1],
            pen="k",
            clear=True,
        )
        self.eeg_plot.addItem(self.line1)
        self.eeg_plot.addItem(self.line2)
        ax1 = self.eeg_plot.getAxis("bottom")
        ax1.setTicks([x_labels.items()])

        # emg plot
        self.emg_plot.plot(
            x=self.x_axis,
            y=self.df["emg"][self.x_axis[0] : self.x_axis[-1] + 1],
            pen="k",
            clear=True,
        )
        ax2 = self.emg_plot.getAxis("bottom")
        ax2.setTicks([x_labels.items()])

        # shading for eeg and emg windows
        [self.plot_shading(i) for i in range(self.window_num)]
        self.eeg_plot.addItem(
            pg.LinearRegionItem(
                [self.window_start, self.window_end],
                movable=False,
                brush=self.color_scheme(self.epoch, center=True),
            )
        )
        self.emg_plot.addItem(
            pg.LinearRegionItem(
                [self.window_start, self.window_end],
                movable=False,
                brush=self.color_scheme(self.epoch, center=True),
            )
        )

        # power spectrum plots
        power_x_axis = np.arange(0, (len(self.eeg_power[self.epoch]) * 0.1), 0.1)
        self.eeg_power_plot.plot(
            x=power_x_axis,
            y=self.eeg_power[self.epoch],
            pen=pg.mkPen("k", width=2),
            clear=True,
        )

        # relative power bar plot remove old values
        self.plot_relative_power()

        # reset hypnogram
        self.hypnogram.removeItem(self.hypno_line)
        self.hypno_line = pg.InfiniteLine(
            pos=self.epoch, movable=True, angle=90, pen=pg.mkPen(width=3, color="r")
        )
        self.hypno_line.sigPositionChangeFinished.connect(self.hypno_go)
        self.hypnogram.addItem(self.hypno_line)
