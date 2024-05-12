from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
from PyQt5.QtCore import *
import sys
import os

import funcs
import utils
from funcs import Buttons


class Tab(QWidget):
    def __init__(self):
        super().__init__()

        # Call update_plots on this instance?
        ## Display.utils.update_plots()?
        self.Display = utils.Display()

    def home(self) -> QWidget:

        epoch = self.Display.epoch

        # Define home as widget and define layout
        outer_layout = QVBoxLayout()
        button_layout = QGridLayout()
        top_layout = QHBoxLayout()
        middle_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()

        # # #  TOP LAYOUT # # #
        # load data button
        button_layout.addWidget(self.Display.load_data_button, 0, 0)

        # load scores button
        load_scores_btn = QPushButton("Load Scores", self)
        # load_scores_btn.clicked.connect(self.load_scores)
        button_layout.addWidget(load_scores_btn, 0, 1)

        # clear scores on current file
        clear_scores_btn = QPushButton("Clear Scores", self)
        # clear_scores_btn.clicked.connect(self.clear_scores)
        button_layout.addWidget(clear_scores_btn, 0, 5)

        # export scores as .txt
        export_scores_btn = QPushButton("Export Scores", self)
        # export_scores_btn.clicked.connect(self.export_scores)
        button_layout.addWidget(export_scores_btn, 0, 4)

        # export score breakdown as .csv
        export_breakdown_btn = QPushButton("Export Breakdown", self)
        # export_breakdown_btn.clicked.connect(self.export_breakdown)
        button_layout.addWidget(export_breakdown_btn, 1, 4)

        # move window to epoch of your choosing
        find_epoch_btn = QPushButton("Find epoch", self)
        # find_epoch_btn.clicked.connect(self.find_epoch)
        button_layout.addWidget(find_epoch_btn, 1, 2)

        # Window size dropdown
        window_size = self.Display.window_size
        window_size.addItems(["5", "1", "3", "7"])
        # self.window_size.currentTextChanged.connect(self.update_plots)
        button_layout.addWidget(window_size, 0, 2)

        # next REM button
        next_REM_btn = QPushButton("Next: {}".format("REM"), self)
        # next_REM_btn.clicked.connect(self.next_rem)
        button_layout.addWidget(next_REM_btn, 0, 3)

        # file name labels
        file_win = self.Display.file_win
        file_win.setText("File: {}".format("N/A"))
        button_layout.addWidget(file_win, 1, 0)

        score_win = self.Display.score_win
        score_win.setText("Scores: {}".format("N/A"))
        button_layout.addWidget(score_win, 1, 1)

        # Current epoch label
        epoch_win = self.Display.epoch_win
        epoch_win.setText("Epoch: {} ".format(epoch))
        button_layout.addWidget(epoch_win, 1, 3)

        # # #  MIDDLE LAYOUT # # #
        # eeg plot
        eeg_plot = self.Display.eeg_plot
        eeg_plot.plot(x=[], y=[])
        top_layout.addWidget(eeg_plot, stretch=3)

        # eeg power spectrum plot
        eeg_power_plot = self.Display.eeg_power_plot
        eeg_power_plot.plot(x=[], y=[])
        top_layout.addWidget(eeg_power_plot, stretch=1)

        # emg plot
        emg_plot = self.Display.emg_plot
        emg_plot.plot(x=[], y=[])
        middle_layout.addWidget(emg_plot, stretch=3)

        # relative power bar chart
        eeg_bar_plot = self.Display.eeg_bar_plot
        middle_layout.addWidget(eeg_bar_plot, stretch=1)

        # # # BOTTOM LAYOUT # # #
        # hypnogram
        hypnogram = self.Display.hypnogram
        hypnogram.plot(x=[], y=[])
        bottom_layout.addWidget(hypnogram)

        # add layouts together
        outer_layout.addLayout(button_layout)
        outer_layout.addLayout(top_layout)
        outer_layout.addLayout(middle_layout)
        outer_layout.addLayout(bottom_layout)
        self.setLayout(outer_layout)

        return self

    def model(self) -> QWidget:
        layout = QGridLayout()

        load_model_btn = QPushButton("Load Model", self)
        layout.addWidget(load_model_btn, 0, 0)

        score_data_btn = QPushButton("Score Data", self)
        layout.addWidget(score_data_btn, 0, 1)

        self.setLayout(layout)

        return self
