from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDateTime
import pyqtgraph as pg
import os
import numpy as np
import pandas as pd


class General(QWidget):
    def __init__(self):
        super().__init__()

        # PATH
        self.current_path = os.path.dirname(os.path.realpath(__file__))

        # Timestamp
        self.timestamp_selected = False
        self.timestamp = None
        self.TIMESTAMP_FORMAT_PLOTTING = "%Y-%m-%d\n%H:%M:%S"
        self.TIMESTAMP_FORMAT_EXPORTING = "%Y-%m-%d %H:%M:%S"

        # Dark Period start and end times
        self.LightDark_input = False
        self.DarkTimeStart = None
        self.DarkTimeEnd = None


class Timestamp_Dialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Start Date and Time of recording (yyyy-MM-dd HH:mm:ss)")
        self.setGeometry(100, 100, 500, 100)

        layout = QVBoxLayout()
        self.dateTimeEdit = QDateTimeEdit(self)
        self.dateTimeEdit.setDateTime(QDateTime.currentDateTime())
        self.dateTimeEdit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        layout.addWidget(self.dateTimeEdit)

        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def getDateTime(self):
        return self.dateTimeEdit.dateTime()


class LightDark_Dialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Start and End Time of Dark periods")
        self.setGeometry(100, 100, 500, 100)

        layout = QVBoxLayout()
        start_time_label = QLabel("Enter start time of first block")
        self.DarkTimeStart = QDateTimeEdit(self)
        self.DarkTimeStart.setDateTime(QDateTime.currentDateTime())
        self.DarkTimeStart.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        layout.addWidget(start_time_label)
        layout.addWidget(self.DarkTimeStart)

        end_time_label = QLabel("Enter end time of first block")
        self.DarkTimeEnd = QDateTimeEdit(self)
        self.DarkTimeEnd.setDateTime(QDateTime.currentDateTime())
        self.DarkTimeEnd.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        layout.addWidget(end_time_label)
        layout.addWidget(self.DarkTimeEnd)

        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def getBlockStartEnd(self):
        self.validateBlockStartEnd()
        return self.DarkTimeStart.dateTime(), self.DarkTimeEnd.dateTime()

    def validateBlockStartEnd(self):
        if (
            not self.DarkTimeStart.dateTime().toPyDateTime()
            < self.DarkTimeEnd.dateTime().toPyDateTime()
        ):
            raise ValueError("Start of dark phase must come before end of dark period")

    def add_block(self):

        # Light/Dark dropdown

        #



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

        self.samplerate = np.NaN  # Number of samples per second
        self.seconds_per_epoch = 10
        self.array_size = np.NaN  # Number of samples per epoch

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
