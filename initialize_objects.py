from PyQt5.QtWidgets import *
import pyqtgraph as pg
import os
import numpy as np
import pandas as pd


class General(QWidget):
    def __init__(self):
        super().__init__()

        # PATH
        self.current_path = os.path.dirname(os.path.realpath(__file__))


class Data(QWidget):
    def __init__(self):
        super().__init__()

        self.epoch = 0
        self.epoch_dict = {}
        self.epoch_list = []

        self.df = pd.DataFrame(columns=["eeg", "emg"])
        self.metrics = pd.DataFrame(
            columns=["delta_rel", "theta_rel", "theta_over_delta"]
        )

        self.samplerate = np.NaN
        self.array_size = np.NaN

        self.window_start = np.NaN
        self.window_end = np.NaN

        self.eeg_power = {}
        self.emg_power = {}

        self.wake_count = 0
        self.nrem_count = 0
        self.rem_count = 0


class Home(QWidget):
    def __init__(self):
        super().__init__()

        # # # TOP LAYOUT # # #

        # Current epoch label
        self.current_epoch_label = QLabel("Epoch: ", self)

        # Current file label
        self.current_wav_file_label = QLabel("File: ", self)

        # Current score label
        self.current_score_file_label = QLabel("Scores: ", self)

        # create plot objects in window
        self.window_size_dropdown = QComboBox(self)
        self.window_size_dropdown.addItems(["1", "3", "5", "7"])
        self.window_size_dropdown.setCurrentIndex(2)

        # Buttons
        self.load_data_button = QPushButton("Load Data", self)
        self.load_scores_button = QPushButton("Load Scores", self)
        self.clear_scores_button = QPushButton("Clear Scores", self)
        self.export_scores_button = QPushButton("Export Scores", self)
        self.export_breakdown_button = QPushButton("Export Breakdown", self)
        self.find_epoch_button = QPushButton("Find epoch", self)
        self.next_REM_button = QPushButton("Next REM", self)

        # # # MIDDLE LAYOUT # # #
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
            pos=0, movable=True, angle=90, pen=pg.mkPen(width=3, color="r")
        )


class Model(QWidget):
    def __init__(self):
        super().__init__()

        # Model labels
        self.current_model_label = QLabel("Current Model Version: None", self)
        self.scoring_complete_label = QLabel("", self)

        # Buttons
        self.load_model_button = QPushButton("Load Model", self)
        self.score_data_button = QPushButton("Score Data", self)

        # Model Objects
        self.model = np.NaN
        self.params = {}
        self.label_encoder = {}