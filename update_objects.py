"""
TODO:
Add methods:
- Key methods to put here are any which directly alter the value of Display() attributes
- Some methods may make sense to put in separate module as standalone functions (calculate_eeg_axis(?))
- This will probably be a huge pain in the ass
"""

from PyQt5.QtWidgets import *
from initialize_objects import Display


class Funcs(QWidget):
    def __init__(self):
        super().__init__()

        self.Display = Display()

    # def update_plots(self):
    #     x_labels = self.Display.calculate_eeg_axes()
    #
    #     # eeg plot
    #     self.eeg_plot.plot(
    #         x=self.x_axis,
    #         y=self.df["eeg"][self.x_axis[0] : self.x_axis[-1] + 1],
    #         pen="k",
    #         clear=True,
    #     )
    #     self.eeg_plot.addItem(self.line1)
    #     self.eeg_plot.addItem(self.line2)
    #     ax1 = self.eeg_plot.getAxis("bottom")
    #     ax1.setTicks([x_labels.items()])
    #
    #     # emg plot
    #     self.emg_plot.plot(
    #         x=self.x_axis,
    #         y=self.df["emg"][self.x_axis[0] : self.x_axis[-1] + 1],
    #         pen="k",
    #         clear=True,
    #     )
    #     ax2 = self.emg_plot.getAxis("bottom")
    #     ax2.setTicks([x_labels.items()])
    #
    #     # shading for eeg and emg windows
    #     [self.plot_shading(i) for i in range(self.window_num)]
    #     self.eeg_plot.addItem(
    #         pg.LinearRegionItem(
    #             [self.window_start, self.window_end],
    #             movable=False,
    #             brush=self.color_scheme(self.epoch, center=True),
    #         )
    #     )
    #     self.emg_plot.addItem(
    #         pg.LinearRegionItem(
    #             [self.window_start, self.window_end],
    #             movable=False,
    #             brush=self.color_scheme(self.epoch, center=True),
    #         )
    #     )
    #
    #     # power spectrum plots
    #     power_x_axis = np.arange(0, (len(self.eeg_power[self.epoch]) * 0.1), 0.1)
    #     self.eeg_power_plot.plot(
    #         x=power_x_axis,
    #         y=self.eeg_power[self.epoch],
    #         pen=pg.mkPen("k", width=2),
    #         clear=True,
    #     )
    #
    #     # relative power bar plot remove old values
    #     self.plot_relative_power()
    #
    #     # reset hypnogram
    #     self.hypnogram.removeItem(self.hypno_line)
    #     self.hypno_line = pg.InfiniteLine(
    #         pos=self.epoch, movable=True, angle=90, pen=pg.mkPen(width=3, color="r")
    #     )
    #     self.hypno_line.sigPositionChangeFinished.connect(self.hypno_go)
    #     self.hypnogram.addItem(self.hypno_line)
