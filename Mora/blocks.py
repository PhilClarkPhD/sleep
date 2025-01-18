from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QDateTimeEdit,
    QComboBox,
    QDialogButtonBox,
    QPushButton,
    QHBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QDateTime


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

        # Button to add more blocks
        self.add_block_button = QPushButton("Add Another Block")
        self.add_block_button.clicked.connect(self.add_block)
        self.layout.addWidget(self.add_block_button)

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

        # Start time
        start_time_label = QLabel("Start Time:")
        start_time_edit = QDateTimeEdit()
        current_time = QDateTime.currentDateTime()
        start_time_edit.setDateTime(current_time)
        start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        details_layout.addWidget(start_time_label)
        details_layout.addWidget(start_time_edit)

        # End time
        end_time_label = QLabel("End Time:")
        end_time_edit = QDateTimeEdit()
        end_time = current_time.addSecs(12 * 3600)  # Add 12 hours in seconds
        end_time_edit.setDateTime(end_time)
        end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        details_layout.addWidget(end_time_label)
        details_layout.addWidget(end_time_edit)

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

    def get_blocks(self):
        # Retrieve data from all blocks
        blocks = []
        for i in range(self.blocks_layout.count()):
            block_widget = self.blocks_layout.itemAt(i).widget()
            if block_widget:
                start_time_edit = block_widget.findChildren(QDateTimeEdit)[0]
                end_time_edit = block_widget.findChildren(QDateTimeEdit)[1]
                dropdown = block_widget.findChildren(QComboBox)[0]

                blocks.append(
                    {
                        "start_time": start_time_edit.dateTime().toPyDateTime(),
                        "end_time": end_time_edit.dateTime().toPyDateTime(),
                        "light_or_dark": dropdown.currentText(),
                    }
                )
        return blocks

    def validate_within_block_timestamps(self):
        # Validate that start time < end time for each block
        for block in self.get_blocks():
            if not block["start_time"] < block["end_time"]:
                raise ValueError("Start time must come before end time for all blocks")

    def validate_no_block_ovelap(self):
        pass

    def validate_block_monotonic_increasing(self):
        pass


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = LightDark_Dialog()

    if dialog.exec() == QDialog.Accepted:
        try:
            dialog.validate_blocks()
            blocks = dialog.get_blocks()
            print("Blocks:", blocks)
        except ValueError as e:
            print("Error:", e)
