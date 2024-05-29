from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from initialize_objects import General, Home, Model, Data
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from data_processing import sleep_functions as sleep
import pyqtgraph as pg
from PyQt5.QtGui import QKeyEvent


class Funcs(QWidget):
    def __init__(self):
        super().__init__()

        self.General = General()
        self.Home = Home()
        self.Model = Model()
        self.Data = Data()

        # brushes for coloring scoring windows
        self.rem = pg.mkBrush(pg.intColor(90, alpha=50))
        self.non_rem = pg.mkBrush(pg.intColor(15, alpha=50))
        self.wake = pg.mkBrush(pg.intColor(20, alpha=70))
        self.unscored = pg.mkBrush(pg.intColor(50, alpha=50))

        self.rem_center = pg.mkBrush(pg.intColor(90, alpha=110))
        self.non_rem_center = pg.mkBrush(pg.intColor(15, alpha=110))
        self.wake_center = pg.mkBrush(pg.intColor(20, alpha=130))
        self.unscored_center = pg.mkBrush(pg.intColor(50, alpha=110))

    def find_epoch(self) -> None:
        num, ok = QInputDialog.getInt(
            self,
            "Find epoch",
            "Enter value between 0 and {}".format(str(len(self.Data.epoch_list) - 1)),
            value=self.Data.epoch,
            min=0,
            max=len(self.Data.epoch_list) - 1,
        )

        if ok:
            self.Data.epoch = num
            self.update_plots()

    def load_scores(self) -> None:
        path, ext = QFileDialog.getOpenFileName(
            self, "Open .txt file", self.General.current_path, "(*.txt *.csv)"
        )

        if path:
            self.General.current_path = os.path.dirname(path)
            self.Home.current_score_file_label.setText(
                "Scores: {}".format(str(os.path.basename(path)))
            )

            score_import = pd.read_csv(path)
            score_import.rename(
                columns=lambda x: x.strip(), inplace=True
            )  # remove whitespace

            if score_import.shape[1] > 2:
                # subtract 1 to force epoch start at 0 when loading scores from Sirenia
                self.Data.epoch_dict = dict(
                    zip(
                        score_import["Epoch #"].values - 1, score_import["Score"].values
                    )
                )
            else:
                self.Data.epoch_dict = dict(
                    zip(score_import["epoch"].values, score_import["score"].values)
                )

            self.update_plots()
            self.plot_hypnogram()

    def name_file(self) -> str:
        get_name = QInputDialog()
        name, ok = get_name.getText(self, "Enter file name", "Enter file name")
        if ok:
            return name
        else:
            raise ValueError("Not a valid input")

    def export_scores(self) -> None:
        name = str(self.name_file())
        path = QFileDialog.getExistingDirectory(
            self, "Select folder", self.General.current_path
        )
        self.General.current_path = path
        file_path = path + "/" + name

        score_export = pd.DataFrame(
            [self.Data.epoch_dict.keys(), self.Data.epoch_dict.values()]
        ).T
        score_export.columns = ["epoch", "score"]
        score_export.to_csv(file_path + ".csv", sep=",", index=False)

    def breakdown_scores(self, epoch: str) -> None:
        if epoch == "Wake":
            self.Data.wake_count += 1
        elif epoch == "Non REM":
            self.Data.nrem_count += 1
        elif epoch == "REM":
            self.Data.rem_count += 1
        else:
            next

    def breakdown_df(self) -> pd.DataFrame:
        [self.breakdown_scores(i) for i in self.Data.epoch_dict.values()]

        total = self.Data.wake_count + self.Data.nrem_count + self.Data.rem_count
        rem_proportion = round(self.Data.rem_count / total, 2)
        nrem_proportion = round(self.Data.nrem_count / total, 2)
        wake_proportion = round(self.Data.wake_count / total, 2)

        labels = ["Wake", "Non REM", "REM"]
        proportions = [wake_proportion, nrem_proportion, rem_proportion]
        totals = [self.Data.wake_count, self.Data.nrem_count, self.Data.rem_count]

        output = pd.DataFrame([labels, totals, proportions]).T
        output.columns = ["Stage", "Epochs", "Proportion"]

        return output

    def export_breakdown(self):
        name = str(self.name_file())
        path = QFileDialog.getExistingDirectory(
            self, "Select folder", self.General.current_path
        )

        self.General.current_path = path
        file_path = path + "/" + name

        breakdown = self.breakdown_df()
        breakdown.to_csv(file_path + ".csv", sep=",", index=False)

    def next_rem(self) -> None:
        for key in range(self.Data.epoch + 1, self.Data.epoch_list[-1]):
            if key not in self.Data.epoch_dict.keys():
                pass
            elif self.Data.epoch_dict[key] == "REM":
                self.Data.epoch = key
                self.plot_hypnogram()
                self.update_plots()
                break
            else:
                pass

    def clear_scores(self) -> None:
        warning_box = QMessageBox()
        warning_box.setIcon(QMessageBox.Warning)
        warning_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        warning_box.setText("Are you sure you want to clear scores?")

        val = warning_box.exec()
        if val == QMessageBox.Ok:
            self.Home.current_score_file_label.setText("Scores: {}".format("N/A"))
            self.Data.epoch_dict = {}
            self.plot_hypnogram()
            self.update_plots()
        else:
            pass

    def color_scheme(self, epoch: int, center: bool = False) -> pg.mkBrush:
        try:
            if self.Data.epoch_dict[epoch] == "REM" and not center:
                return self.rem
            elif self.Data.epoch_dict[epoch] == "REM" and center:
                return self.rem_center
            elif self.Data.epoch_dict[epoch] == "Non REM" and not center:
                return self.non_rem
            elif self.Data.epoch_dict[epoch] == "Non REM" and center:
                return self.non_rem_center
            elif self.Data.epoch_dict[epoch] == "Wake" and not center:
                return self.wake
            elif self.Data.epoch_dict[epoch] == "Wake" and center:
                return self.wake_center
            elif self.Data.epoch_dict[epoch] == "Unscored" and center:
                return self.unscored_center
            elif self.Data.epoch_dict[epoch] == [] and center:
                return self.unscored_center
            elif center:
                return self.unscored_center
            else:
                return self.unscored

        except KeyError:
            return self.unscored

    def load_data(self) -> None:
        path, ext = QFileDialog.getOpenFileName(
            self, "Open .wav file", self.General.current_path, "(*.wav)"
        )

        if path:
            self.General.current_path = os.path.dirname(path)
            self.Home.current_wav_file_label.setText(
                "File: {}".format(str(os.path.basename(path)))
            )

            self.Data.samplerate, data = wavfile.read(path)
            self.Data.array_size = self.Data.samplerate * 10
            self.Data.df = pd.DataFrame(data=data, columns=["eeg", "emg"])

            # Compute power spectrum, relative power metrics
            eeg_power, emg_power = sleep.compute_power(
                self.Data.df, samplerate=self.Data.samplerate
            )
            smoothed_eeg, smoothed_emg = sleep.smooth_signal(eeg_power, emg_power)
            relative_power = sleep.compute_relative_power(smoothed_eeg)

            self.Data.metrics = relative_power[
                ["delta_rel", "theta_rel", "theta_over_delta"]
            ]
            self.Data.epoch_list = relative_power.index
            self.Data.eeg_power = smoothed_eeg
            self.Data.emg_power = smoothed_emg
            self.Data.delta_vals = self.Data.metrics[["delta_rel"]].values
            self.Data.theta_vals = self.Data.metrics[["theta_rel"]].values

            self.update_plots()
            self.plot_hypnogram()

    def plot_shading(self, i: int) -> None:
        begin = self.x_axis[0] + i * self.Data.array_size
        end = begin + self.Data.array_size

        self.Home.eeg_plot.addItem(
            pg.LinearRegionItem(
                [begin, end],
                movable=False,
                brush=self.color_scheme(begin / self.Data.array_size),
            )
        )
        self.Home.emg_plot.addItem(
            pg.LinearRegionItem(
                [begin, end],
                movable=False,
                brush=self.color_scheme(begin / self.Data.array_size),
            )
        )

    def calculate_eeg_axes(self):
        # x axis for eeg/emg plots
        current_window_number = int(self.Home.window_size_dropdown.currentText())
        self.Home.current_epoch_label.setText(f"Epoch: {self.Data.epoch}")

        # calculate start and end points for x axis in current window
        self.Data.window_start = self.Data.epoch * self.Data.array_size
        self.Data.window_end = self.Data.window_start + self.Data.array_size

        # modify x axis according to user-selected window size
        x_start = int(
            self.Data.window_start
            - (((current_window_number - 1) / 2) * self.Data.array_size)
        )
        x_end = int(
            self.Data.window_end
            + (((current_window_number - 1) / 2) * self.Data.array_size)
        )

        # limit x axis to begin at 0
        if x_start < 0:
            x_end -= x_start
            x_start = 0

        self.x_axis = np.arange(x_start, x_end)

        # convert x values to time
        x_labels = np.linspace(
            (x_start / self.Data.samplerate),
            (x_end / self.Data.samplerate),
            current_window_number + 1,
        )
        x_anchors = np.linspace(x_start, x_end, current_window_number + 1)
        x_labels = dict(zip(x_anchors, x_labels))

        # Deal with mismatch in window length and data length at end of dataframe
        if len(self.x_axis) != len(self.Data.df["eeg"][x_start:x_end]):
            self.x_axis = np.arange(self.x_axis[0], len(self.Data.df))

        return x_labels

    def plot_relative_power(self) -> None:
        self.Home.eeg_bar_plot.removeItem(self.Home.rel_delta)
        self.Home.eeg_bar_plot.removeItem(self.Home.rel_theta)

        self.Home.rel_delta = pg.BarGraphItem(
            x=[0],
            height=self.Data.metrics["delta_rel"].values[self.Data.epoch],
            width=0.8,
            brush=self.non_rem_center,
        )
        self.Home.rel_theta = pg.BarGraphItem(
            x=[1],
            height=self.Data.metrics["theta_rel"].values[self.Data.epoch],
            width=0.8,
            brush=self.rem_center,
        )

        self.Home.eeg_bar_plot.addItem(self.Home.rel_delta)
        self.Home.eeg_bar_plot.addItem(self.Home.rel_theta)

    @staticmethod
    def convert_to_numbers(x: str) -> int:
        if x == "Wake":
            return 0
        if x == "Non REM":
            return 1
        if x == "REM":
            return 2
        if x == "Unscored":
            return 3
        else:
            return np.NaN

    def plot_hypnogram(self) -> pg.plot:  # TODO: This plot does not render for 171N_bl
        hypno_list = [self.convert_to_numbers(x) for x in self.Data.epoch_dict.values()]
        epoch_list = list(self.Data.epoch_dict.keys())

        return self.Home.hypnogram.plot(
            x=epoch_list, y=hypno_list, pen=pg.mkPen("k", width=2), clear=True
        )

    def hypno_go(self) -> None:
        self.Data.epoch = round(self.Home.hypno_line.value())
        self.update_plots()

    def update_plots(self):
        x_labels = self.calculate_eeg_axes()

        # eeg plot
        self.Home.eeg_plot.plot(
            x=self.x_axis,
            y=self.Data.df["eeg"][self.x_axis[0] : self.x_axis[-1] + 1],
            pen="k",
            clear=True,
        )

        self.Home.eeg_plot.addItem(self.Home.line1)
        self.Home.eeg_plot.addItem(self.Home.line2)
        ax1 = self.Home.eeg_plot.getAxis("bottom")
        ax1.setTicks([x_labels.items()])

        # emg plot
        self.Home.emg_plot.plot(
            x=self.x_axis,
            y=self.Data.df["emg"][self.x_axis[0] : self.x_axis[-1] + 1],
            pen="k",
            clear=True,
        )
        ax2 = self.Home.emg_plot.getAxis("bottom")
        ax2.setTicks([x_labels.items()])

        # shading for eeg and emg windows
        current_window_number = int(self.Home.window_size_dropdown.currentText())
        [self.plot_shading(i) for i in range(current_window_number)]
        self.Home.eeg_plot.addItem(
            pg.LinearRegionItem(
                [self.Data.window_start, self.Data.window_end],
                movable=False,
                brush=self.color_scheme(self.Data.epoch, center=True),
            )
        )
        self.Home.emg_plot.addItem(
            pg.LinearRegionItem(
                [self.Data.window_start, self.Data.window_end],
                movable=False,
                brush=self.color_scheme(self.Data.epoch, center=True),
            )
        )

        # power spectrum plots
        power_x_axis = np.arange(
            0, (len(self.Data.eeg_power[self.Data.epoch]) * 0.1), 0.1
        )
        self.Home.eeg_power_plot.plot(
            x=power_x_axis,
            y=self.Data.eeg_power[self.Data.epoch],
            pen=pg.mkPen("k", width=2),
            clear=True,
        )

        # relative power bar plot remove old values
        self.plot_relative_power()

        # reset hypnogram
        self.Home.hypnogram.removeItem(self.Home.hypno_line)
        self.Home.hypno_line = pg.InfiniteLine(
            pos=self.Data.epoch,
            movable=True,
            angle=90,
            pen=pg.mkPen(width=3, color="r"),
        )
        self.Home.hypno_line.sigPositionChangeFinished.connect(self.hypno_go)
        self.Home.hypnogram.addItem(self.Home.hypno_line)

    def check_epoch(self, movement: int) -> None:
        new_epoch = self.Data.epoch + movement

        if new_epoch not in self.Data.epoch_list:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Critical)
            error_box.setText("Not a valid epoch")
            error_box.exec_()
            raise KeyError("Not a valid epoch")
        else:
            self.Data.epoch += movement
            self.plot_hypnogram()
            self.update_plots()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Left:
            self.check_epoch(movement=-1)

        if event.key() == Qt.Key_Right:
            self.check_epoch(movement=1)

        if event.key() == Qt.Key_W:  # Wake
            self.Data.epoch_dict[self.Data.epoch] = "Wake"
            self.check_epoch(movement=1)

        if event.key() == Qt.Key_E:  # Non REM
            self.Data.epoch_dict[self.Data.epoch] = "Non REM"
            self.check_epoch(movement=1)

        if event.key() == Qt.Key_R:  # REM
            self.Data.epoch_dict[self.Data.epoch] = "REM"
            self.check_epoch(movement=1)

        if event.key() == Qt.Key_T:  # Unscored
            self.Data.epoch_dict[self.Data.epoch] = "Unscored"
            self.check_epoch(movement=1)

    def load_model(self) -> None:
        import joblib

        model_path, *_ = QFileDialog.getOpenFileName(
            self, "Open .pkl file", self.General.current_path, "(*.joblib *.pkl)"
        )

        # Reset current path
        self.General.current_path = os.path.dirname(model_path)

        # Get model and params from model_path
        model, params, label_encoder = joblib.load(model_path)

        # Set class attribute values and display
        self.Model.model = model
        self.Model.params = params
        self.Model.label_encoder = label_encoder
        self.Model.current_model_label.setText(
            "Current Model Version: {}".format(self.Model.params["model_version"])
        )
        self.Model.scoring_complete_label.setText("")

    def score_data(self) -> None:

        num, ok = QInputDialog.getInt(
            self,
            "Enter Epoch Number",
            "Select representative epoch",
            min=0,
            max=len(self.Data.epoch_list) - 1,
            value=2,
        )

        if ok:
            self.Data.epoch = num

        self.Data.metrics = sleep.generate_features(self.Data.df, self.Data.epoch)

        features = self.Data.metrics[
            [
                "EEG_std",
                "EEG_ss",
                "EEG_amp",
                "EMG_std",
                "EMG_ss",
                "EMG_events",
                "delta_rel",
                "theta_rel",
                "theta_over_delta",
            ]
        ]

        predictions = self.Model.model.predict(features)
        decoded_predictions = self.Model.label_encoder.inverse_transform(predictions)
        self.Data.scores = decoded_predictions

        # temporarily deprecating modify_scores until we figure out the best way to implement it
        # scores = sleep.modify_scores(decoded_predictions)

        self.Data.epoch_dict = dict(zip(self.Data.epoch_list, self.Data.scores))
        self.update_plots()
        self.plot_hypnogram()
        self.Model.scoring_complete_label.setText("Scoring Complete!")
