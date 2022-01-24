# my modules
from NN.optimizer.sgd import SGD
from hw1_utils import resource_path, DatasetManager, get2DInfos, get3DInfos
from NN.network import Network
from NN.layer import DenseLayer, Tanh, Softmax
from NN.loss import CategoricalCrossEntropy

# libraries
import typing
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigurCanvas
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QFileDialog
from pathlib import Path
import sys
import os
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")


# load main ui from resource to class
# 用abs path讓pyinstaller將resouce包在exe檔案內
main_ui_path = resource_path("./hw1_qtui/main.ui")
main_ui_class, main_window_class = uic.loadUiType(main_ui_path)


class MyDatasetCanvas(FigurCanvas):
    def __init__(self, width=4.71, height=4.21, dpi=100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = None
        super().__init__(self.fig)

    def plotDataset(self, X, y, show_line, network: Network, dataset_name):
        self.fig.clear()
        if X is None or len(X) == 0:
            self.draw()
            return

        try:
            dim = 3 if len(X[0]) >= 3 else len(X[0])
        except TypeError as e:
            dim = 0

        if dim == 3:
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.ax.clear()

            self.ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=5)
            self.ax.set_title(f"dataset : {dataset_name}")
            self.ax.set_xlabel("dim1")
            self.ax.set_ylabel("dim2")
            self.ax.set_zlabel("dim3")
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.clear()
            if dim == 2:
                self.ax.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=10)
            else:
                self.ax.scatter(X, np.zeros(len(X)), c=y, s=10)

            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")

        if show_line > 0 and network is not None:
            perceptronLayer = network.layerStack[0]
            if dim == 2:
                x1_min, x1_max, x2_min, x2_max, F2D = get2DInfos(X, perceptronLayer)
                self.ax.set_title(dataset_name)
                dif1 = (x1_max - x1_min) * 0.05
                dif2 = (x2_max - x2_min) * 0.05
                self.ax.set_ylim(x2_min - dif2, x2_max + dif2)
                x1_line = np.linspace(x1_min - dif1 * 0.05, x1_max + dif1 * 0.05, 100)

                for i in range(len(perceptronLayer.W[0])):
                    x2_value = F2D(x1_line, i)
                    self.ax.plot(x1_line, x2_value)
            elif dim == 3:
                x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, F3D = get3DInfos(X, perceptronLayer)

                dif1 = (x1_max - x1_min) * 0.05
                dif2 = (x2_max - x2_min) * 0.05
                dif3 = (x3_max - x3_min) * 0.05

                x1_line = np.linspace(x1_min - dif1, x1_max + dif1, 20)
                x2_line = np.linspace(x2_min - dif2, x2_max + dif2, 20)
                x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)
                for i in range(len(perceptronLayer.W[0])):
                    x3_value = F3D(x1_mesh, x2_mesh, i)

                    if show_line == 1:
                        self.ax.plot_surface(x1_mesh, x2_mesh, x3_value, color=(0.3 * i, 0.5, 0.3))
                    else:
                        self.ax.plot_wireframe(x1_mesh, x2_mesh, x3_value, color=(0.3 * i, 0.5, 0.3))

                self.ax.set_zlim(x3_min - dif3, x3_max + dif3)

        self.draw()


class MyMainWindow(main_window_class):
    def __init__(self, main_ui_class) -> None:
        super().__init__()
        self.main_ui = main_ui_class()
        self.main_ui.setupUi(self)

        # initialize
        self.dataset_manager = DatasetManager()
        self.MyNet = None

        self.setUpExtraWidgets()
        self.setUpUiEvent()

    def setUpExtraWidgets(self):
        self.canvas = MyDatasetCanvas()
        self.canvas.setParent(self.main_ui.figure_placeholder)

    def setUpUiEvent(self):
        self.main_ui.select_file_btn.clicked.connect(self.QselectFileDialog)
        self.main_ui.show_train_data_cbox.stateChanged.connect(self.updateUi)
        self.main_ui.show_test_data_cbox.stateChanged.connect(self.updateUi)
        self.main_ui.show_lines_cbox.stateChanged.connect(self.updateUi)
        self.main_ui.show_wires_cbox.stateChanged.connect(self.updateUi)

        self.main_ui.normalized_cbox.stateChanged.connect(self.change_data_format)
        self.main_ui.pca_dim_slider.valueChanged[int].connect(self.change_data_format)

        self.main_ui.start_train_btn.clicked.connect(self.startTraining)
        self.main_ui.start_test_btn.clicked.connect(self.startTesting)
        self.main_ui.pca_dim_slider.setValue(0)

    def change_data_format(self):
        self.createEmptyNet()
        self.updateUi()

    def startTesting(self):
        if self.MyNet is None:
            return

        test_x, test_y = self.getXy(get_test=True)
        if test_x is None:
            return

        predictions = self.MyNet.Predict(test_x).argmax(axis=1)
        answers = test_y.argmax(axis=1)

        self.main_ui.show_test_accuracy_label.setText(str(self.MyNet.getAccuracy(predictions, answers)))

    def startTraining(self):
        if self.MyNet is None:
            return

        learning_rate = self.main_ui.learning_rate_spin_box.value()
        self.MyNet.setOptimizer(SGD(learning_rate))
        epochs = self.main_ui.epochs_spin_box.value()
        batchsize = self.main_ui.batchsize_spin_box.value()

        train_x, train_y = self.getXy(get_train=True)
        if train_x is None:
            return

        for e in range(epochs):
            total_err = 0
            for bx, by in self.MyNet.getNextRandomBatch(train_x, train_y, batchsize):
                total_err += self.MyNet.train_on_batch(bx, by)

            total_batches = (len(train_x) + batchsize - 1) // batchsize
            self.main_ui.show_loss_label.setText(str(total_err / total_batches if total_batches > 0 else 0))

        predictions = self.MyNet.Predict(train_x).argmax(axis=1)
        answers = train_y.argmax(axis=1)
        self.main_ui.show_train_accuracy_label.setText(str(self.MyNet.getAccuracy(predictions, answers)))

        self.updateUi()

    def QselectFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, "選擇資料庫檔案", os.getcwd(), "Text files (*.txt *.csv *.tsv)")

        filepath = Path(fname[0])
        if filepath.is_file():
            self.load_dataset(filepath)

    def load_dataset(self, dataset_path):
        self.dataset_manager.load_dataset(dataset_path, split_train_test=True, split_ratio=0.33333)

        self.createEmptyNet()
        self.updateUi()

    def createEmptyNet(self):
        if self.dataset_manager.dataset_path == None:
            self.MyNet = None
            return

        train_x, train_y = self.getXy(get_train=True)
        if train_x is None:
            self.MyNet = None
            return

        self.layer1_line = 4
        self.layer2_line = 8
        dataset_dim = len(train_x[0])
        self.MyNet = Network(
            [
                DenseLayer(dataset_dim, self.layer1_line),
                Tanh(),
                DenseLayer(self.layer1_line, self.layer2_line),
                Tanh(),
                DenseLayer(self.layer2_line, len(train_y[0])),
                Softmax(),
            ],
            optimizer=SGD(self.main_ui.learning_rate_spin_box.value()),  # specify learning rate of sgd
            loss_func=CategoricalCrossEntropy(),
        )

        self.main_ui.show_loss_label.setText("Not trained yet")
        self.main_ui.show_train_accuracy_label.setText("Not trained yet")
        self.main_ui.show_test_accuracy_label.setText("Not trained yet")

    def getXy(self, get_train=False, get_test=False):
        # get normalized or PCAed data from dataset manager
        normalized = self.main_ui.normalized_cbox.isChecked()
        pca_num = self.main_ui.pca_dim_slider.value()

        # show train or test data
        if get_train:
            train_x, train_y = self.dataset_manager.getXy(
                is_test=False, normalized=normalized, onehot=True, pca_num=pca_num
            )
        else:
            train_x, train_y = None, None

        if get_test:
            test_x, test_y = self.dataset_manager.getXy(
                is_test=True, normalized=normalized, onehot=True, pca_num=pca_num
            )
        else:
            test_x, test_y = None, None

        if train_x is not None and test_x is not None:
            x = np.concatenate([train_x, test_x])
            y = np.concatenate([train_y, test_y])
        elif train_x is None and test_x is not None:
            x, y = test_x, test_y
        elif test_x is None and train_x is not None:
            x, y = train_x, train_y
        else:
            x, y = None, None
        return x, y

    def updateUi(self):
        self.main_ui.show_pca_dim_label.setText(str(self.main_ui.pca_dim_slider.value()))

        if self.dataset_manager.dataset_path == None:
            return

        self.main_ui.selected_file_label.setText(self.dataset_manager.dataset_name)

        get_train = self.main_ui.show_train_data_cbox.isChecked()
        get_test = self.main_ui.show_test_data_cbox.isChecked()
        x, y = self.getXy(get_train, get_test)

        show_line = 0
        if self.main_ui.show_lines_cbox.isChecked():
            show_line = 1
            if self.main_ui.show_wires_cbox.isChecked():
                show_line = 2

        self.canvas.plotDataset(
            x, y.argmax(axis=1) if y is not None else y, show_line, self.MyNet, self.dataset_manager.dataset_name
        )


class MyHw1App(QApplication):
    def __init__(self, argv: typing.List[str]) -> None:
        super().__init__(argv)

        # create ui and window obj after app created
        self.main_window = MyMainWindow(main_ui_class)

    def start(self):
        self.main_window.show()
        self.exec()


if __name__ == "__main__":
    myApp = MyHw1App(sys.argv)
    myApp.start()
