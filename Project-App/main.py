import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QTableWidget,
                             QTableWidgetItem, QTextEdit, QComboBox, QSpinBox)
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np


class BreastCancerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breast Cancer datasæt udforskning")
        self.setGeometry(100, 100, 1000, 800)

        # Load the dataset
        self.breast_cancer = load_breast_cancer()
        self.df = pd.DataFrame(data=self.breast_cancer['data'],
                               columns=self.breast_cancer['feature_names'])
        self.df['target'] = self.breast_cancer['target']

        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        title = QLabel("<h1>Breast Cancer Datasæt Udforskning.</h1>")
        title.setStyleSheet("margin: 10px;")
        main_layout.addWidget(title)

        control_panel = QWidget()
        control_layout = QHBoxLayout()

        self.load_button = QPushButton("Vis Hele Datasættet")
        self.load_button.clicked.connect(self.show_full_dataset)

        self.head_button = QPushButton("Vis 5 Første Rækker")
        self.head_button.clicked.connect(self.show_head)

        self.shuffle_button = QPushButton("Shuffle Datasættet")
        self.shuffle_button.clicked.connect(self.shuffle_dataset)

        self.stats_button = QPushButton("Vis Statistik")
        self.stats_button.clicked.connect(self.show_statistics)

        self.feature_combo = QComboBox()
        self.feature_combo.addItems(self.df.columns[:-1])  # Exclude target column
        self.feature_combo.currentIndexChanged.connect(self.feature_selected)

        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, len(self.df))
        self.sample_spin.setValue(5)

        control_layout.addWidget(QLabel("Feature:"))
        control_layout.addWidget(self.feature_combo)
        control_layout.addWidget(QLabel("Sample Size:"))
        control_layout.addWidget(self.sample_spin)
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.head_button)
        control_layout.addWidget(self.shuffle_button)
        control_layout.addWidget(self.stats_button)

        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)

        self.table = QTableWidget()
        main_layout.addWidget(self.table)

        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setMaximumHeight(150)
        main_layout.addWidget(self.info_display)

        self.statusBar().showMessage(f"Dataset loaded with {len(self.df)} rows and {len(self.df.columns)} columns")

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.show_head()

    def show_full_dataset(self):
        self.display_data(self.df)

    def show_head(self):
        self.display_data(self.df.head())

    def shuffle_dataset(self):
        sample_size = self.sample_spin.value()
        if sample_size == len(self.df):
            shuffled_df = self.df.sample(frac=1).reset_index(drop=True)
        else:
            shuffled_df = self.df.sample(n=sample_size)
        self.display_data(shuffled_df)
        self.info_display.append(f"Shuffled and showing {len(shuffled_df)} samples")

    def show_statistics(self):
        selected_feature = self.feature_combo.currentText()
        if selected_feature:
            stats = self.df[selected_feature].describe()
            stats_text = f"Statistics for {selected_feature}:\n"
            stats_text += f"Count: {stats['count']}\n"
            stats_text += f"Mean: {stats['mean']:.4f}\n"
            stats_text += f"Std: {stats['std']:.4f}\n"
            stats_text += f"Min: {stats['min']:.4f}\n"
            stats_text += f"25%: {stats['25%']:.4f}\n"
            stats_text += f"50%: {stats['50%']:.4f}\n"
            stats_text += f"75%: {stats['75%']:.4f}\n"
            stats_text += f"Max: {stats['max']:.4f}"

            self.info_display.clear()
            self.info_display.append(stats_text)

            full_stats = self.df.describe()
            self.display_data(full_stats)

    def feature_selected(self):
        selected_feature = self.feature_combo.currentText()
        if selected_feature:
            mean_value = self.df[selected_feature].mean()
            self.info_display.clear()
            self.info_display.append(f"Selected feature: {selected_feature}")
            self.info_display.append(f"Mean value: {mean_value:.4f}")

            # Show the selected feature column
            self.display_data(self.df[[selected_feature, 'target']].head(10))

    def display_data(self, data):
        if isinstance(data, pd.DataFrame):
            self.table.setRowCount(data.shape[0])
            self.table.setColumnCount(data.shape[1])
            self.table.setHorizontalHeaderLabels(data.columns)

            for row in range(data.shape[0]):
                for col in range(data.shape[1]):
                    item = QTableWidgetItem(str(data.iloc[row, col]))
                    self.table.setItem(row, col, item)

            self.table.resizeColumnsToContents()

            # Update status bar
            self.statusBar().showMessage(f"Displaying {data.shape[0]} rows and {data.shape[1]} columns")


def main():
    app = QApplication(sys.argv)
    window = BreastCancerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()