"""
TODO:
1. refactor similar to fp_gui
    a. make sure computed metrics match those of model
    b. add in necessary metrics in the model tab
3. make 'NEXT' button selectable ['NREM','WAKE','REM','Unscored']
5. Modeling
    a. allow file browsing to select model (.joblib file)
    b. display key metrics of model performance on test set (accuracy, conf matrix, f1, jaccard, etc.)
    c. Allow training and saving of new model from current data/score set
    d. Allow training/selection of best model?
    e. Set up system for comparing two set of scores)

"""

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, \
    QFileDialog, QLabel, QInputDialog, QMessageBox, QTabWidget, QVBoxLayout, \
    QGridLayout, QComboBox, QHBoxLayout
from PyQt5.QtGui import QKeyEvent
import sys
from scipy.io import wavfile
import pandas as pd
import numpy as np
import sleep_functions as sleep
import os
from pycaret.classification import *

pg.setConfigOption('background', "w")
pg.setConfigOption('foreground', "k")

# starting path
START_PATH = os.path.dirname(os.path.realpath(__file__))


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(50, 50, 1125, 525)
        self.setWindowTitle('Mora Sleep Analysis')
        self.setStyleSheet("font-size: 20px;")

        # path variables
        self.current_path = START_PATH

        # error box for selecting epoch out of range
        self.error_box = QMessageBox()
        self.error_box.setIcon(QMessageBox.Critical)

        # warning box for clearing scores
        self.warning = QMessageBox()
        self.warning.setIcon(QMessageBox.Warning)
        self.warning.setText('Are you sure you want to clear scores?')
        self.warning.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        # Model labels
        self.model_display = QLabel('Current_Model: None', self)
        self.scoring_complete = QLabel('', self)

        # data objects
        self.epoch_dict = {}
        self.df = pd.DataFrame(columns=['eeg', 'emg'])
        self.metrics = pd.DataFrame(columns=['delta_rel', 'theta_rel', 'theta_over_delta'])
        self.samplerate = np.NaN
        self.epoch = 0
        self.window_num = 5
        self.array_size = np.NaN
        self.window_start = np.NaN
        self.window_end = np.NaN
        self.epoch_list = []
        self.eeg_power = {}
        self.emg_power = {}
        self.eeg_y = []
        self.emg_y = []

        # create plot objects in window
        self.window_size = QComboBox(self)
        self.eeg_plot = pg.PlotWidget(self, title='EEG')
        self.line1 = pg.InfiniteLine(pos=80, movable=True, angle=0, pen=pg.mkPen(width=3, color='r'))
        self.line2 = pg.InfiniteLine(pos=-80, movable=True, angle=0, pen=pg.mkPen(width=3, color='r'))
        self.emg_plot = pg.PlotWidget(self, title='EMG')
        self.eeg_power_plot = pg.PlotWidget(self, title='Power Spectrum')
        self.eeg_bar_plot = pg.PlotWidget(self, title='Relative Power')

        self.rel_delta = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.rel_theta = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.labels = {0: 'delta', 1: 'theta'}
        (self.eeg_bar_plot.getAxis('bottom')).setTicks([self.labels.items()])

        self.hypnogram = pg.PlotWidget(self, title='Hypnogram')
        self.hyp_labels = {0: 'Wake', 1: 'Non REM', 2: 'REM', 3: 'Unscored'}
        (self.hypnogram.getAxis('left')).setTicks([self.hyp_labels.items()])
        self.hypno_line = pg.InfiniteLine(pos=self.epoch, movable=True, angle=90, pen=pg.mkPen(width=3, color='r'))

        # brushes for coloring scoring windows
        self.rem = pg.mkBrush(pg.intColor(90, alpha=50))
        self.non_rem = pg.mkBrush(pg.intColor(15, alpha=50))
        self.wake = pg.mkBrush(pg.intColor(20, alpha=70))
        self.unscored = pg.mkBrush(pg.intColor(50, alpha=50))

        self.rem_center = pg.mkBrush(pg.intColor(90, alpha=110))
        self.non_rem_center = pg.mkBrush(pg.intColor(15, alpha=110))
        self.wake_center = pg.mkBrush(pg.intColor(20, alpha=130))
        self.unscored_center = pg.mkBrush(pg.intColor(50, alpha=110))

        # epoch label
        self.epoch_win = QLabel("Epoch: {} ".format(self.epoch), self)

        # current file label
        self.file_win = QLabel('', self)

        # current score label
        self.score_win = QLabel('', self)

        # axes for raw eeg/emg plots
        self.x_axis = []

        # y axis for eeg barplot
        self.delta_vals = self.metrics[['delta_rel']].values
        self.theta_vals = self.metrics[['theta_rel']].values

        # Model objects
        self.model = np.NaN
        self.scores = np.NaN

        # Set Layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        tabs = QTabWidget()
        tabs.addTab(self.home(), 'Home')
        tabs.addTab(self.model_tab(), 'Model')

        layout.addWidget(tabs)

    def home(self) -> QWidget:

        # Define home as widget and define layout
        home = QWidget()
        outer_layout = QVBoxLayout()
        button_layout = QGridLayout()
        top_layout = QHBoxLayout()
        middle_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()

        # Define buttons and add to top layout
        # load data button
        load_data_btn = QPushButton('Load Data', self)
        load_data_btn.clicked.connect(self.load_data)
        button_layout.addWidget(load_data_btn, 0, 0)

        # load scores button
        load_scores_btn = QPushButton('Load Scores', self)
        load_scores_btn.clicked.connect(self.load_scores)
        button_layout.addWidget(load_scores_btn, 0, 1)

        # clear scores on current file
        clear_scores_btn = QPushButton('Clear Scores', self)
        clear_scores_btn.clicked.connect(self.clear_scores)
        button_layout.addWidget(clear_scores_btn, 0, 5)

        # export scores as .txt
        export_scores_btn = QPushButton('Export Scores', self)
        export_scores_btn.clicked.connect(self.export_scores)
        button_layout.addWidget(export_scores_btn, 0, 4)

        # move window to epoch of your choosing
        find_epoch_btn = QPushButton('Find epoch', self)
        find_epoch_btn.clicked.connect(self.find_epoch)
        button_layout.addWidget(find_epoch_btn, 1, 2)

        # Window size dropdown
        self.window_size.addItems(['5', '1', '3', '7'])
        self.window_size.currentTextChanged.connect(self.update_plots)
        button_layout.addWidget(self.window_size, 0, 2)

        next_REM_btn = QPushButton('Next: {}'.format(''), self)
        next_REM_btn.clicked.connect(self.next_rem)
        button_layout.addWidget(next_REM_btn, 0, 3)

        # file name labels
        self.file_win.setText('File: {}'.format('N/A'))
        button_layout.addWidget(self.file_win, 1, 0)

        self.score_win.setText('Scores: {}'.format('N/A'))
        button_layout.addWidget(self.score_win, 1, 1)

        # Current epoch label
        self.epoch_win.setText("Epoch: {} ".format(self.epoch))
        button_layout.addWidget(self.epoch_win, 1, 3)

        # Plots
        # eeg plot
        self.eeg_plot.plot(x=[], y=[])
        top_layout.addWidget(self.eeg_plot, stretch=3)

        # eeg power spectrum plot
        self.eeg_power_plot.plot(x=[], y=[])
        top_layout.addWidget(self.eeg_power_plot, stretch=1)

        # emg plot
        self.emg_plot.plot(x=[], y=[])
        middle_layout.addWidget(self.emg_plot, stretch=3)

        # relative power bar chart
        middle_layout.addWidget(self.eeg_bar_plot, stretch=1)

        # hypnogram
        self.hypnogram.plot(x=[], y=[])
        bottom_layout.addWidget(self.hypnogram)

        # add layouts together
        outer_layout.addLayout(button_layout)
        outer_layout.addLayout(top_layout)
        outer_layout.addLayout(middle_layout)
        outer_layout.addLayout(bottom_layout)
        home.setLayout(outer_layout)

        return home

    def find_epoch(self) -> None:
        num, ok = QInputDialog.getInt(self,
                                      'Find epoch',
                                      'Enter value between 0 and {}'.format(str(len(self.epoch_list) - 1)),
                                      value=self.epoch,
                                      min=0,
                                      max=len(self.epoch_list) - 1)

        if ok:
            self.epoch = num
            self.update_plots()

    def load_data(self) -> None:
        path, ext = QFileDialog.getOpenFileName(self, 'Open .wav file', self.current_path, '(*.wav)')

        if path:
            self.current_path = os.path.dirname(path)
            self.file_win.setText('File: {}'.format(str(os.path.basename(path))))

            self.samplerate, data = wavfile.read(path)
            self.array_size = self.samplerate * 10
            self.df = pd.DataFrame(data=data, columns=['eeg', 'emg'])
            self.eeg_y = self.df['eeg']
            self.emg_y = self.df['emg']

            # Compute power spectrum, relative power metrics
            eeg_power, emg_power = sleep.compute_power(self.df, samplerate=self.samplerate)
            smoothed_eeg, smoothed_emg = sleep.smooth_signal(eeg_power, emg_power)
            relative_power = sleep.compute_relative_power(smoothed_eeg)

            self.metrics = relative_power[['delta_rel', 'theta_rel', 'theta_over_delta']]
            self.epoch_list = relative_power.index
            self.eeg_power = smoothed_eeg
            self.emg_power = smoothed_emg
            self.delta_vals = self.metrics[['delta_rel']].values
            self.theta_vals = self.metrics[['theta_rel']].values
            self.update_plots()

    def load_scores(self) -> None:
        path, ext = QFileDialog.getOpenFileName(self, 'Open .txt file', self.current_path, '(*.txt)')

        if path:
            self.current_path = os.path.dirname(path)
            self.score_win.setText('Scores: {}'.format(str(os.path.basename(path))))

            score_import = pd.read_csv(path)
            if score_import.shape[1] > 2:
                # -1 to force epoch start at 0 when loading scores from Sirenia
                self.epoch_dict = dict(zip(score_import['Epoch #'].values - 1, score_import[' Score'].values))
            else:
                self.epoch_dict = dict(
                    zip(score_import['epoch'].values, score_import['score'].values))

            self.update_plots()
            self.plot_hypnogram()

    def name_file(self) -> str:
        get_name = QInputDialog()
        name, ok = get_name.getText(self, 'Enter file name', 'Enter file name')
        if ok:
            return name
        else:
            raise ValueError('Not a valid input')

    def export_scores(self) -> None:
        name = str(self.name_file())
        file = QFileDialog.getExistingDirectory(self, 'Select folder', self.current_path)
        file_path = file + '/' + name

        score_export = pd.DataFrame([self.epoch_dict.keys(), self.epoch_dict.values()]).T
        score_export.columns = ['epoch', 'score']
        score_export.to_csv(file_path + '.txt', sep=',', index=False)

    def next_rem(self) -> None:
        for key in range(self.epoch + 1, self.epoch_list[-1]):
            if key not in self.epoch_dict.keys():
                pass
            elif self.epoch_dict[key] == 2:
                self.epoch = key
                self.plot_hypnogram()
                self.update_plots()
                break
            else:
                pass

    def clear_scores(self) -> None:
        val = self.warning.exec()
        if val == QMessageBox.Ok:
            self.score_win.setText('Scores: {}'.format('N/A'))
            self.epoch_dict = {}
            self.plot_hypnogram()
            self.update_plots()
        else:
            pass

    def check_epoch(self, modifier: int) -> None:
        new_epoch = self.epoch + modifier
        if new_epoch not in range(0, len(self.epoch_list)):
            self.error_box.setText('Not a valid epoch')
            self.error_box.exec_()
            raise KeyError('Not a valid epoch')

    def calculate_eeg_axes(self) -> dict:
        # x axis for eeg/emg plots
        self.window_num = int(self.window_size.currentText())
        self.epoch_win.setText("Epoch: {} ".format(self.epoch))

        # calculate start and end points for x axis in current window
        self.window_start = self.epoch * self.array_size
        self.window_end = self.window_start + self.array_size

        # modify x axis according to user-selected window size
        x_start = int(self.window_start - (((self.window_num - 1) / 2) * self.array_size))
        x_end = int(self.window_end + (((self.window_num - 1) / 2) * self.array_size))

        # limit x axis to begin at 0
        if x_start < 0:
            x_end -= x_start
            x_start = 0

        self.x_axis = np.arange(x_start, x_end)

        # convert x values to time
        x_labels = np.linspace((x_start / self.samplerate), (x_end / self.samplerate),
                               self.window_num + 1)
        x_anchors = np.linspace(x_start, x_end, self.window_num + 1)
        x_labels = dict(zip(x_anchors, x_labels))

        # Deal with mismatch in window length and data length at end of dataframe
        if len(self.x_axis) != len(self.df['eeg'][x_start:x_end]):
            self.x_axis = np.arange(self.x_axis[0], len(self.df))

        return x_labels

    def plot_shading(self, i: int) -> None:
        begin = self.x_axis[0] + i * self.array_size
        end = begin + self.array_size

        self.eeg_plot.addItem(pg.LinearRegionItem([begin, end], movable=False,
                                                  brush=self.color_scheme(begin / self.array_size)))
        self.emg_plot.addItem(pg.LinearRegionItem([begin, end], movable=False,
                                                  brush=self.color_scheme(begin / self.array_size)))

    def plot_relative_power(self) -> None:
        self.eeg_bar_plot.removeItem(self.rel_delta)
        self.eeg_bar_plot.removeItem(self.rel_theta)

        self.rel_delta = pg.BarGraphItem(x=[0], height=self.delta_vals[self.epoch], width=0.8,
                                         brush=self.non_rem_center)
        self.rel_theta = pg.BarGraphItem(x=[1], height=self.theta_vals[self.epoch], width=0.8,
                                         brush=self.rem_center)

        self.eeg_bar_plot.addItem(self.rel_delta)
        self.eeg_bar_plot.addItem(self.rel_theta)

    def update_plots(self) -> None:
        x_labels = self.calculate_eeg_axes()

        # eeg plot
        self.eeg_plot.plot(x=self.x_axis, y=self.df['eeg'][self.x_axis[0]:self.x_axis[-1] + 1], pen='k', clear=True)
        self.eeg_plot.addItem(self.line1)
        self.eeg_plot.addItem(self.line2)
        ax1 = self.eeg_plot.getAxis('bottom')
        ax1.setTicks([x_labels.items()])

        # emg plot
        self.emg_plot.plot(x=self.x_axis, y=self.df['emg'][self.x_axis[0]:self.x_axis[-1] + 1], pen='k', clear=True)
        ax2 = self.emg_plot.getAxis('bottom')
        ax2.setTicks([x_labels.items()])

        # shading for eeg and emg windows
        [self.plot_shading(i) for i in range(self.window_num)]
        self.eeg_plot.addItem(pg.LinearRegionItem([self.window_start, self.window_end], movable=False,
                                                  brush=self.color_scheme(self.epoch, center=True)))
        self.emg_plot.addItem(pg.LinearRegionItem([self.window_start, self.window_end], movable=False,
                                                  brush=self.color_scheme(self.epoch, center=True)))

        # power spectrum plots
        power_x_axis = np.arange(0, (len(self.eeg_power[self.epoch]) * 0.1), 0.1)
        self.eeg_power_plot.plot(x=power_x_axis, y=self.eeg_power[self.epoch], pen=pg.mkPen('k', width=2), clear=True)

        # relative power bar plot remove old values
        self.plot_relative_power()

        # reset hypnogram
        self.hypnogram.removeItem(self.hypno_line)
        self.hypno_line = pg.InfiniteLine(pos=self.epoch, movable=True, angle=90, pen=pg.mkPen(width=3, color='r'))
        self.hypno_line.sigPositionChangeFinished.connect(self.hypno_go)
        self.hypnogram.addItem(self.hypno_line)

    @staticmethod
    def convert_to_numbers(x: str) -> int:
        if x == 'Wake':
            return 0
        if x == 'Non REM':
            return 1
        if x == 'REM':
            return 2
        if x == 'Unscored':
            return 3
        else:
            return np.NaN

    def plot_hypnogram(self) -> pg.plot:
        hypno_list = [self.convert_to_numbers(x) for x in self.epoch_dict.values()]
        return self.hypnogram.plot(x=list(self.epoch_dict.keys()), y=hypno_list,
                                   pen=pg.mkPen('k', width=2), clear=True)

    def hypno_go(self) -> None:
        self.epoch = round(self.hypno_line.value())
        self.update_plots()

    def color_scheme(self, epoch: int, center: bool = False) -> pg.mkBrush:
        try:
            if self.epoch_dict[epoch] == 'REM' and not center:
                return self.rem
            elif self.epoch_dict[epoch] == 'REM' and center:
                return self.rem_center
            elif self.epoch_dict[epoch] == 'Non REM' and not center:
                return self.non_rem
            elif self.epoch_dict[epoch] == 'Non REM' and center:
                return self.non_rem_center
            elif self.epoch_dict[epoch] == 'Wake' and not center:
                return self.wake
            elif self.epoch_dict[epoch] == 'Wake' and center:
                return self.wake_center
            elif self.epoch_dict[epoch] == 'Unscored' and center:
                return self.unscored_center
            elif self.epoch_dict[epoch] == [] and center:
                return self.unscored_center
            elif center:
                return self.unscored_center
            else:
                return self.unscored

        except KeyError:
            return self.unscored

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == 87:
            self.epoch_dict[self.epoch] = 'Wake'
            # 'Wake'
            self.check_epoch(1)
            self.epoch += 1
            self.plot_hypnogram()
            self.update_plots()

        if event.key() == 69:
            self.epoch_dict[self.epoch] = 'Non REM'
            # 'Non REM'
            self.check_epoch(1)
            self.epoch += 1
            self.plot_hypnogram()
            self.update_plots()

        if event.key() == 82:
            self.epoch_dict[self.epoch] = 'REM'
            # 'REM'
            self.check_epoch(1)
            self.epoch += 1
            self.plot_hypnogram()
            self.update_plots()

        if event.key() == 84:
            self.epoch_dict[self.epoch] = 'Unscored'
            # 'Unscored'
            self.check_epoch(1)
            self.epoch += 1
            self.plot_hypnogram()
            self.update_plots()

        if event.key() == 16777234:
            # left arrow
            self.check_epoch(-1)
            self.epoch -= 1
            self.update_plots()

        if event.key() == 16777236:
            # right arrow
            self.check_epoch(1)
            self.epoch += 1
            self.update_plots()

    def model_tab(self) -> QWidget:
        model_tab = QWidget()

        layout = QGridLayout()

        load_model_btn = QPushButton('Load Model', self)
        load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(load_model_btn, 0, 0)

        score_data_btn = QPushButton('Score Data', self)
        score_data_btn.clicked.connect(self.score_data)
        layout.addWidget(score_data_btn, 0, 1)

        layout.addWidget(self.model_display, 1, 0)
        layout.addWidget(self.scoring_complete, 1, 1)

        model_tab.setLayout(layout)

        return model_tab

    def load_model(self) -> None:
        """
        TODO: pickle current model and use that - the .joblib here is old news
        :return: None
        """
        file = QFileDialog.getOpenFileName(self, 'Open .wav file', self.current_path, '(*.joblib *.pkl)')
        file_path = file[0]
        self.current_path = os.path.dirname(file_path)
        file = os.path.splitext(file_path)[0]

        self.model = load_model(file)
        self.model_display.setText('Current_Model: {}'.format(str(os.path.basename(file_path))))
        self.scoring_complete.setText('')

    def score_data(self) -> None:

        num, ok = QInputDialog.getInt(self,
                                      'Enter Epoch Number',
                                      'Select representative REM epoch',
                                      min=0,
                                      max=len(self.epoch_list) - 1)

        if ok:
            self.epoch = num

        self.metrics = sleep.generate_features(self.df, self.epoch)
        features = self.metrics[['EEG_std', 'EEG_amp', 'EMG_std', 'delta_rel', 'theta_rel', 'theta_over_delta']]

        predictions = predict_model(self.model, features)
        self.scores = sleep.modify_scores(predictions['Label'].values)

        self.epoch_dict = dict(zip(self.epoch_list, self.scores))
        self.update_plots()
        self.plot_hypnogram()
        self.scoring_complete.setText('Scoring Complete!')


def run():
    app = QApplication(sys.argv)
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run()
