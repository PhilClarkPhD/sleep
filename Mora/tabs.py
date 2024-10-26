from PyQt5.QtWidgets import *
from update_objects import Funcs


class Tabs(QWidget):
    def __init__(self):
        super().__init__()

        self.Funcs = Funcs()

    def home_tab(self) -> QWidget:

        home_tab = QWidget()

        # Define layouts
        outer_layout = QVBoxLayout()
        button_layout = QGridLayout()
        top_layout = QHBoxLayout()
        middle_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()

        # # #  TOP LAYOUT # # #
        # epoch label
        current_epoch_label = self.Funcs.Home.current_epoch_label
        button_layout.addWidget(current_epoch_label, 1, 3)

        # load data button
        load_data_button = self.Funcs.Home.load_data_button
        load_data_button.clicked.connect(self.Funcs.load_data)
        button_layout.addWidget(load_data_button, 0, 0)

        # load scores button
        load_scores_button = self.Funcs.Home.load_scores_button
        load_scores_button.clicked.connect(self.Funcs.load_scores)
        button_layout.addWidget(load_scores_button, 0, 1)

        # clear scores on current file
        clear_scores_button = self.Funcs.Home.clear_scores_button
        clear_scores_button.clicked.connect(self.Funcs.clear_scores)
        button_layout.addWidget(clear_scores_button, 0, 5)

        # export scores as .txt
        export_scores_button = self.Funcs.Home.export_scores_button
        export_scores_button.clicked.connect(self.Funcs.export_scores)
        button_layout.addWidget(export_scores_button, 0, 4)

        # export score breakdown as .csv
        export_breakdown_button = self.Funcs.Home.export_breakdown_button
        export_breakdown_button.clicked.connect(self.Funcs.export_breakdown)
        button_layout.addWidget(export_breakdown_button, 1, 4)

        # move window to epoch of your choosing
        find_epoch_button = self.Funcs.Home.find_epoch_button
        find_epoch_button.clicked.connect(self.Funcs.find_epoch)
        button_layout.addWidget(find_epoch_button, 1, 2)

        # Window size dropdown
        window_size_dropdown = self.Funcs.Home.window_size_dropdown
        window_size_dropdown.currentTextChanged.connect(self.Funcs.update_plots)
        button_layout.addWidget(window_size_dropdown, 0, 2)

        # next REM button
        next_REM_button = self.Funcs.Home.next_REM_button
        next_REM_button.clicked.connect(self.Funcs.next_rem)
        button_layout.addWidget(next_REM_button, 0, 3)

        # File name labels
        current_wav_file_label = self.Funcs.Home.current_wav_file_label
        button_layout.addWidget(current_wav_file_label, 1, 0)

        current_score_file_label = self.Funcs.Home.current_score_file_label
        button_layout.addWidget(current_score_file_label, 1, 1)

        # Current epoch label
        current_epoch_label = self.Funcs.Home.current_epoch_label
        button_layout.addWidget(current_epoch_label, 1, 3)

        # # #  MIDDLE LAYOUT # # #
        # eeg plot
        eeg_plot = self.Funcs.Home.eeg_plot
        eeg_plot.plot(x=[], y=[])
        top_layout.addWidget(eeg_plot, stretch=3)

        # eeg power spectrum plot
        eeg_power_plot = self.Funcs.Home.eeg_power_plot
        eeg_power_plot.plot(x=[], y=[])
        top_layout.addWidget(eeg_power_plot, stretch=1)

        # emg plot
        emg_plot = self.Funcs.Home.emg_plot
        emg_plot.plot(x=[], y=[])
        middle_layout.addWidget(emg_plot, stretch=3)

        # relative power bar chart
        eeg_bar_plot = self.Funcs.Home.eeg_bar_plot
        middle_layout.addWidget(eeg_bar_plot, stretch=1)

        # # # BOTTOM LAYOUT # # #
        # hypnogram
        hypnogram = self.Funcs.Home.hypnogram
        hypnogram.plot(x=[], y=[])
        bottom_layout.addWidget(hypnogram)

        # add layouts together
        outer_layout.addLayout(button_layout)
        outer_layout.addLayout(top_layout)
        outer_layout.addLayout(middle_layout)
        outer_layout.addLayout(bottom_layout)
        home_tab.setLayout(outer_layout)

        return home_tab

    def model_tab(self) -> QWidget:
        model_tab = QWidget()
        layout = QGridLayout()

        # Buttons
        load_model_button = self.Funcs.Model.load_model_button
        load_model_button.clicked.connect(self.Funcs.load_model)
        layout.addWidget(load_model_button, 0, 0)

        score_data_button = self.Funcs.Model.score_data_button
        score_data_button.clicked.connect(self.Funcs.score_data)
        layout.addWidget(score_data_button, 0, 1)

        # Labels
        current_model_label = self.Funcs.Model.current_model_label
        layout.addWidget(current_model_label, 1, 0)

        scoring_complete_label = self.Funcs.Model.scoring_complete_label
        layout.addWidget(scoring_complete_label, 1, 1)

        model_tab.setLayout(layout)

        return model_tab
