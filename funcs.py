from PyQt5.QtWidgets import *
from scipy.io import wavfile
import pandas as pd
from data_processing import sleep_functions as sleep
import os


class General(QWidget):
    def __init__(self):
        super().__init__()

        self.current_path = os.path.dirname(os.path.realpath(__file__))


class Buttons(General):
    def __init__(self):
        super().__init__()

    def load_data(self) -> None:
        from utils import Display

        Display = Display()

        path, ext = QFileDialog.getOpenFileName(
            self, "Open .wav file", self.current_path, "(*.wav)"
        )

        if path:
            self.current_path = os.path.dirname(path)
            Display.file_win.setText(
                "File: {}".format(str(os.path.basename(path)))
            )  # Not updating in window!
            print(Display.file_win.text())

            # self.samplerate, data = wavfile.read(path)
            # self.array_size = self.samplerate * 10
            # self.df = pd.DataFrame(data=data, columns=["eeg", "emg"])
            #
            # # Compute power spectrum, relative power metrics
            # eeg_power, emg_power = sleep.compute_power(
            #     self.df, samplerate=self.samplerate
            # )
            # smoothed_eeg, smoothed_emg = sleep.smooth_signal(eeg_power, emg_power)
            # relative_power = sleep.compute_relative_power(smoothed_eeg)
            #
            # self.metrics = relative_power[
            #     ["delta_rel", "theta_rel", "theta_over_delta"]
            # ]
            # self.epoch_list = relative_power.index
            # self.eeg_power = smoothed_eeg
            # self.emg_power = smoothed_emg
            # self.delta_vals = self.metrics[["delta_rel"]].values
            # self.theta_vals = self.metrics[["theta_rel"]].values
            # self.update_plots()
