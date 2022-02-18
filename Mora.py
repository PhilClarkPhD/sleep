'''
TODO:
1. refactor similar to fp_gui
3. make 'NEXT' button selectable ['NREM','WAKE','REM','Unscored']
5. Modeling
    a. allow file browsing to select model (.joblib file)
    b. display key metrics of model performance on test set (accuracy, conf matrix, f1, jaccard, etc.)
    c. Allow training and saving of new model from current data/score set
    d. Allow training/selection of best model?
    e. Set up system for comparing two set of scores)

'''

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, \
    QFileDialog, QLabel, QInputDialog, QMessageBox, QTabWidget, QVBoxLayout, \
    QGridLayout, QCheckBox, QComboBox, QHBoxLayout
import sys
from scipy.io import wavfile
import pandas as pd
import numpy as np
import sleep_functions as sleep
from joblib import dump, load
import os

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
        self.epoch_list = []
        self.eeg_power = {}
        self.emg_power = {}
        self.eeg_y = []
        self.emg_y = []

        # plot objects
        self.window_size = QComboBox(self)
        self.window_num = 5
        self.window_start = self.epoch * 10000
        self.window_end = self.window_start + 10000
        self.eeg_plot = pg.PlotWidget(self, title='EEG')
        self.line1 = pg.InfiniteLine(pos=5, movable=True, angle=0, pen=pg.mkPen(width=3, color='r'))
        self.line2 = pg.InfiniteLine(pos=-5, movable=True, angle=0, pen=pg.mkPen(width=3, color='r'))
        self.emg_plot = pg.PlotWidget(self, title='EMG')
        self.eeg_power_plot = pg.PlotWidget(self, title='Power Spectrum')
        self.eeg_bar_plot = pg.PlotWidget(self, title='Relative Power')

        self.rel_delta = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.rel_theta = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.labels = {0: 'delta', 1: 'theta'}
        self.ax = self.eeg_bar_plot.getAxis('bottom')
        self.ax.setTicks([self.labels.items()])

        self.hypnogram = pg.PlotWidget(self, title='Hypnogram')
        self.hypno_y = []
        self.hyp_labels = {0: 'Wake', 1: 'Non REM', 2: 'REM', 3: 'Unscored'}
        self.hypno_y_ax = self.hypnogram.getAxis('left')
        self.hypno_y_ax.setTicks([self.hyp_labels.items()])
        self.hypno_line = pg.InfiniteLine(pos=self.epoch, movable=True, angle=90, pen=pg.mkPen(width=3, color='r'))

        # brushes for coloring scoring windows
        self.rem = pg.mkBrush(pg.intColor(90, alpha=50))
        self.non_rem = pg.mkBrush(pg.intColor(15, alpha=50))
        self.wake = pg.mkBrush(pg.intColor(20, alpha=70))
        self.unscored = pg.mkBrush(pg.intColor(50, alpha=50))

        self.rem_center = pg.mkBrush(pg.intColor(90, alpha=50))
        self.non_rem_center = pg.mkBrush(pg.intColor(15, alpha=50))
        self.wake_center = pg.mkBrush(pg.intColor(20, alpha=70))
        self.unscored_center = pg.mkBrush(pg.intColor(50, alpha=50))

        # epoch label
        self.epoch_win = QLabel("Epoch: {} ".format(self.epoch), self)

        # current file label
        self.file_win = QLabel('', self)

        # current score label
        self.score_win = QLabel('', self)

        # axes for raw eeg/emg plots
        self.raw_y_start = []
        self.raw_y_end = []
        self.raw_x = []

        # axes for power spectrum plots
        self.eeg_power_y = []
        self.emg_power_y = []
        self.power_x = []

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

    def home(self):

        # Define home as widget and define layout
        home = QWidget()
        outer_layout = QVBoxLayout()
        button_layout = QGridLayout()
        top_layout = QHBoxLayout()
        middle_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()

        ## Define buttons and add to top layout
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

    def find_epoch(self):
        num, ok = QInputDialog.getInt(self,
                                      'Find epoch',
                                      'Enter value between 0 and {}'.format(str(len(self.epoch_list) - 1)),
                                      value=self.epoch,
                                      min=0,
                                      max=len(self.epoch_list) - 1)

        if ok:
            self.epoch = num
            self.update_plots()

    def load_data(self):
        file = QFileDialog.getOpenFileName(self, 'Open .wav file', self.current_path, '(*.wav)')
        file_path = file[0]
        self.current_path = os.path.dirname(file_path)
        self.file_win.setText('File: {}'.format(str(os.path.basename(file_path))))

        self.samplerate, data = wavfile.read(file_path)
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
        self.update_plots()
        self.hypnogram_func()

    def load_scores(self):
        file = QFileDialog.getOpenFileName(self, 'Open .txt file', self.current_path, '(*.txt)')
        file_path = file[0]
        self.current_path = os.path.dirname(file_path)
        self.score_win.setText('Scores: {}'.format(str(os.path.basename(file_path))))

        score_import = pd.read_csv(file_path)
        if score_import.shape[1] > 2:
            self.epoch_dict = dict(zip(score_import['Epoch #'].values - 1, map(self.convert_to_numbers, score_import[
                ' Score'].values)))  # -1 to force epoch start at 0
        else:
            self.epoch_dict = dict(
                zip(score_import['epoch'].values, map(self.convert_to_numbers, score_import['score'].values)))

        self.update_plots()
        self.hypnogram_func()

    def name_file(self):
        get_name = QInputDialog()
        name, ok = get_name.getText(self, 'Enter file name', 'Enter file name')
        if ok:
            return name
        else:
            raise ValueError('Not a valid input')

    def export_scores(self):
        name = str(self.name_file())
        file = QFileDialog.getExistingDirectory(self, 'Select folder', self.current_path)
        file_path = file + '/' + name

        converted_scores = map(self.convert_to_scores, self.epoch_dict.values())

        score_export = pd.DataFrame([self.epoch_dict.keys(), converted_scores]).T
        score_export.columns = ['epoch', 'score']
        score_export.to_csv(file_path + '.txt', sep=',', index=False)

    def next_rem(self):
        for key in range(self.epoch + 1, self.epoch_list[-1]):
            if key not in self.epoch_dict.keys():
                pass
            elif self.epoch_dict[key] == 2:
                self.epoch = key
                self.hypnogram_func()
                self.update_plots()
                break
            else:
                pass

    def clear_scores(self):
        val = self.warning.exec()
        if val == QMessageBox.Ok:
            self.score_win.setText('Scores: {}'.format('N/A'))
            self.epoch_dict = {}
            self.hypnogram_func()
            self.update_plots()
        else:
            pass

    def check_epoch(self, modifier: int) -> None:
        new_epoch = self.epoch + modifier
        if new_epoch not in range(0, len(self.epoch_list)):
            self.error_box.setText('Not a valid epoch')
            self.error_box.exec_()
            raise KeyError('Not a valid epoch')

    def update_plots(self):

        # Update window number and epoch label
        self.window_num = int(self.window_size.currentText())
        self.epoch_win.setText("Epoch: {} ".format(self.epoch))

        # x and y axes for indices for raw eeg/emg plots
        self.window_start = self.epoch * 10000
        self.window_end = self.window_start + 10000
        self.raw_y_start = int(self.window_start - (((self.window_num - 1) / 2) * 10000))
        self.raw_y_end = int(self.window_end + (((self.window_num - 1) / 2) * 10000))
        self.raw_x = np.arange(self.raw_y_start, self.raw_y_end)

        # Force x axes to start at 0
        if self.raw_y_start < 0:
            self.raw_y_end -= self.raw_y_start
            self.raw_y_start = 0

        # x axis values
        self.raw_x = np.arange(self.raw_y_start, self.raw_y_end)

        # convert x-vals to time (seconds) for display
        x_labels = np.linspace((self.raw_y_start / self.samplerate), (self.raw_y_end / self.samplerate),
                               self.window_num + 1)
        x_anchors = np.linspace(self.raw_x[0], self.raw_x[-1], self.window_num + 1)
        x_labels = dict(zip(x_anchors, x_labels))

        # Deal with mismatch in window length and data length at end of dataframe
        if len(self.raw_x) != len(self.df['eeg'][self.raw_y_start:self.raw_y_end]):
            self.raw_x = np.arange(self.raw_y_start, len(self.df))

        # eeg plot
        self.eeg_plot.plot(x=self.raw_x, y=self.df['eeg'][self.raw_y_start:self.raw_y_end], pen='k', clear=True)
        self.eeg_plot.addItem(self.line1)
        self.eeg_plot.addItem(self.line2)
        ax1 = self.eeg_plot.getAxis('bottom')
        ax1.setTicks([x_labels.items()])

        # emg plot
        self.emg_plot.plot(x=self.raw_x, y=self.df['emg'][self.raw_y_start:self.raw_y_end], pen='k', clear=True)
        ax2 = self.emg_plot.getAxis('bottom')
        ax2.setTicks([x_labels.items()])

        # shading for eeg and emg windows
        for i in range(self.window_num):
            begin = self.raw_y_start + i * 10000
            end = begin + 10000
            self.eeg_plot.addItem(pg.LinearRegionItem([begin, end], movable=False,
                                                      brush=self.color_scheme(begin / 10000)))
            self.eeg_plot.addItem(pg.LinearRegionItem([self.window_start, self.window_end], movable=False,
                                                      brush=self.color_scheme(self.epoch, center=True)))

            self.emg_plot.addItem(pg.LinearRegionItem([begin, end], movable=False,
                                                      brush=self.color_scheme(begin / 10000)))
            self.emg_plot.addItem(pg.LinearRegionItem([self.window_start, self.window_end], movable=False,
                                                      brush=self.color_scheme(self.epoch, center=True)))

        # power spectrum plots
        self.power_x = np.arange(0, (len(self.eeg_power[self.epoch]) * 0.1), 0.1)
        self.eeg_power_plot.plot(x=self.power_x, y=self.eeg_power[self.epoch], pen=pg.mkPen('k', width=2), clear=True)

        # relative power bar plot get_values
        self.delta_vals = self.metrics[['delta_rel']].values
        self.theta_vals = self.metrics[['theta_rel']].values

        # relative power bar plot remove old values
        self.eeg_bar_plot.removeItem(self.rel_delta)
        self.eeg_bar_plot.removeItem(self.rel_theta)

        # relative power bar plot, add new values and re-graph
        self.rel_delta = pg.BarGraphItem(x=[0], height=self.delta_vals[self.epoch], width=0.8,
                                         brush=self.non_rem_center)

        self.rel_theta = pg.BarGraphItem(x=[1], height=self.theta_vals[self.epoch], width=0.8,
                                         brush=self.rem_center)

        self.eeg_bar_plot.addItem(self.rel_delta)
        self.eeg_bar_plot.addItem(self.rel_theta)

        # reset hypnogram
        self.hypnogram.removeItem(self.hypno_line)
        self.hypno_line = pg.InfiniteLine(pos=self.epoch, movable=True, angle=90, pen=pg.mkPen(width=3, color='r'))
        self.hypno_line.sigPositionChangeFinished.connect(self.hypno_go)
        self.hypnogram.addItem(self.hypno_line)

    @staticmethod
    def convert_to_scores(x: int) -> str:
        if x == 0:
            return 'Wake'
        if x == 1:
            return 'Non REM'
        if x == 2:
            return 'REM'
        if x == 3:
            return 'Unscored'
        else:
            return np.NaN

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
            return None

    def hypnogram_func(self):
        return self.hypnogram.plot(x=list(self.epoch_dict.keys()), y=list(self.epoch_dict.values()),
                                   pen=pg.mkPen('k', width=2), clear=True)

    def hypno_go(self):
        current_epoch = self.hypno_line.value()
        self.epoch = round(current_epoch)
        return self.update_plots()

    def color_scheme(self, epoch: int, center: bool = False):
        try:
            if self.epoch_dict[epoch] == 2 and not center:
                return self.rem
            elif self.epoch_dict[epoch] == 2 and center:
                return self.rem_center
            elif self.epoch_dict[epoch] == 1 and not center:
                return self.non_rem
            elif self.epoch_dict[epoch] == 1 and center:
                return self.non_rem_center
            elif self.epoch_dict[epoch] == 0 and not center:
                return self.wake
            elif self.epoch_dict[epoch] == 0 and center:
                return self.wake_center
            elif self.epoch_dict[epoch] == 3 and center:
                return self.unscored_center
            elif self.epoch_dict[epoch] == [] and center:
                return self.unscored_center
            elif center:
                return self.unscored_center
            else:
                return self.unscored

        except KeyError:
            return self.unscored

    def keyPressEvent(self, event: int):
        if event.key() == 87:
            self.epoch_dict[self.epoch] = 0
            # 'Wake'
            self.check_epoch(1)
            self.epoch += 1
            self.hypnogram_func()
            self.update_plots()

        if event.key() == 69:
            self.epoch_dict[self.epoch] = 1
            # 'Non REM'
            self.check_epoch(1)
            self.epoch += 1
            self.hypnogram_func()
            self.update_plots()

        if event.key() == 82:
            self.epoch_dict[self.epoch] = 2
            # 'REM'
            self.check_epoch(1)
            self.epoch += 1
            self.hypnogram_func()
            self.update_plots()

        if event.key() == 84:
            self.epoch_dict[self.epoch] = 3
            # 'Unscored'
            self.check_epoch(1)
            self.epoch += 1
            self.hypnogram_func()
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

    def model_tab(self):
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

    def load_model(self):
        file = QFileDialog.getOpenFileName(self, 'Open .wav file', self.current_path, '(*.joblib *.pkl)')
        file_path = file[0]
        self.current_path = os.path.dirname(file_path)

        self.model = load(os.path.realpath(file_path))
        self.model_display.setText('Current_Model: {}'.format(str(os.path.basename(file_path))))
        self.scoring_complete.setText('')

    def score_data(self):

        try:
            self.scores = self.model.predict(self.metrics.values)
            self.scores = map(self.convert_to_numbers, self.scores)

            self.epoch_dict = dict(zip(self.epoch_list, self.scores))
            self.update_plots()
            self.hypnogram_func()
            self.scoring_complete.setText('Scoring Complete!')

        except ValueError:
            self.error_box.setText('Must load data first!')
            self.error_box.exec_()


def run():
    app = QApplication(sys.argv)
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run()
