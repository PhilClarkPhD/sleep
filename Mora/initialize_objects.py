import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDateTime
import pyqtgraph as pg
import os
import numpy as np
import pandas as pd
from blocks import TimeBlocks, TimeBlockUnit


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

        # Light/Dark start and end times
        self.LightDarkLabels = None


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

        self.setWindowTitle("Start and End Time of Light/Dark Periods")
        self.setGeometry(100, 100, 500, 300)

        self.layout = QVBoxLayout()

        # Container for blocks
        self.blocks_layout = QVBoxLayout()
        self.layout.addLayout(self.blocks_layout)

        # Add the first 3 blocks initially
        self.add_block()
        self.add_block()
        self.add_block()

        #  Add block
        self.add_block_button = QPushButton("Add Block")
        self.add_block_button.clicked.connect(self.add_block)
        self.layout.addWidget(self.add_block_button)

        # Remove block
        self.remove_block_button = QPushButton("Remove Block")
        self.remove_block_button.clicked.connect(self.remove_block)
        self.layout.addWidget(self.remove_block_button)

        # Dialog buttons
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

        self.setLayout(self.layout)

    def add_block(self):
        # Determine block number
        block_number = self.blocks_layout.count() + 1

        # Create a widget for the block
        block_widget = QWidget()
        block_layout = QVBoxLayout()

        # Add block number label at the top
        block_label = QLabel(f"<b>Block {block_number}</b>")
        block_layout.addWidget(block_label)

        # Create horizontal layout for block details
        details_layout = QHBoxLayout()

        # Duration inputs: Hours, Minutes, Seconds
        duration_label = QLabel("Duration (H:M:S):")
        hours_spinbox = QSpinBox()
        hours_spinbox.setRange(0, 23)  # Allow up to 23 hours
        hours_spinbox.setSuffix(" h")
        hours_spinbox.setValue(0)

        minutes_spinbox = QSpinBox()
        minutes_spinbox.setRange(0, 59)  # Allow up to 59 minutes
        minutes_spinbox.setSuffix(" m")
        minutes_spinbox.setValue(0)

        seconds_spinbox = QSpinBox()
        seconds_spinbox.setRange(0, 59)  # Allow up to 59 seconds
        seconds_spinbox.setSuffix(" s")
        seconds_spinbox.setValue(0)

        details_layout.addWidget(duration_label)
        details_layout.addWidget(hours_spinbox)
        details_layout.addWidget(minutes_spinbox)
        details_layout.addWidget(seconds_spinbox)

        # Light/Dark dropdown
        dropdown_label = QLabel("Phase:")
        dropdown = QComboBox()
        dropdown.addItems(["Light", "Dark"])
        details_layout.addWidget(dropdown_label)
        details_layout.addWidget(dropdown)

        # Add the details layout to the block layout below the title
        block_layout.addLayout(details_layout)

        # Set layout for the block widget and add it to the main layout
        block_widget.setLayout(block_layout)
        self.blocks_layout.addWidget(block_widget)

    def remove_block(self):
        if self.blocks_layout.count() > 0:
            last_block = self.blocks_layout.itemAt(
                self.blocks_layout.count() - 1
            ).widget()
            if last_block:
                self.blocks_layout.removeWidget(last_block)
                last_block.deleteLater()

    def get_blocks(self):
        # Retrieve data from all blocks
        blocks = {}
        for i in range(self.blocks_layout.count()):
            block_widget = self.blocks_layout.itemAt(i).widget()
            if block_widget:
                hours = block_widget.findChildren(QSpinBox)[0].value()
                minutes = block_widget.findChildren(QSpinBox)[1].value()
                seconds = block_widget.findChildren(QSpinBox)[2].value()

                duration = datetime.timedelta(
                    hours=hours, minutes=minutes, seconds=seconds
                )

                dropdown = block_widget.findChildren(QComboBox)[0]
                phase = dropdown.currentText()

                block = TimeBlockUnit(duration=duration, phase=phase)

                blocks[i] = block
        return TimeBlocks(blocks=blocks)

    def on_ok_clicked(self):
        try:
            self.get_blocks()
            self.accept()
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", str(e))


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


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = LightDark_Dialog()

    if dialog.exec() == QDialog.Accepted:
        try:
            blocks = dialog.get_blocks()
            print("Blocks:", blocks)
        except ValueError as e:
            print("Error:", e)
