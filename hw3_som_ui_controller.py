from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigurCanvas
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog
import os
import numpy as np

from hw3_som_utils import DatasetManager, resource_path
from NN import SOMnetwork

# load main ui from resource to class
# 用abs path讓pyinstaller將resouce包在exe檔案內
main_ui_path = resource_path('./hw3_som_qtui/main.ui')
main_ui_class, main_window_class = uic.loadUiType(main_ui_path)

class MyDatasetCanvas(FigurCanvas):
    def __init__(self, width=4.71, height=4.21, dpi=100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = None
        super().__init__(self.fig)

    
    def plotDatasetInfo(self, dataset_name, dim):

        if dim == 3:
            self.ax.set_title(f'dataset : {dataset_name}')
            self.ax.set_xlabel('dim1')
            self.ax.set_ylabel('dim2')
            self.ax.set_zlabel('dim3')

        else:
            self.ax.set_title(f'dataset : {dataset_name}')
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')


    def plotNetwork(self, som, dim, show_centroid_edges):
        width = som.width
        # test by showing square of nodes
        # W = np.concatenate([np.array([ni % width for ni in range(som.n_units)]).reshape(1, som.n_units).T, np.array([ni // som.height for ni in range(som.n_units)]).reshape(1, som.n_units).T], axis=1)
        W = som.W

        if dim>=2:
            
            pos_list = []
            for d in range(dim):
                pos_list.append(W[:, d])
            self.ax.scatter(*pos_list, c='blue', s=20)
            
            if show_centroid_edges:
                for wi in range(0, som.n_units, width):
                    pos_list = []
                    for d in range(dim):
                        pos_list.append(W[wi:wi+width, d])
                    self.ax.plot(*pos_list, c='purple')

                for hi in range(width):
                    pos_list = []
                    for d in range(dim):
                        pos_list.append(W[hi:som.n_units:width, d])
                    self.ax.plot(*pos_list, c='purple')
        elif dim==1:
            pos_list = [W[:, 0], np.zeros(som.n_units)]
            self.ax.scatter(*pos_list, c='blue', s=20)

            if show_centroid_edges:
                for wi in range(0, som.n_units, width):
                    show_nodes = W[wi:wi+width, 0]
                    pos_list = [show_nodes, [0]*len(show_nodes)]
                    self.ax.plot(*pos_list, c='purple')

                for hi in range(width):
                    show_nodes = W[hi:som.n_units:width, 0]
                    pos_list = [show_nodes, [0]*len(show_nodes)]
                    self.ax.plot(*pos_list, c='purple')


    def plotNodes(self, X, y, dim):
        if dim >= 3:
            self.ax.scatter(X[:, 0],X[:, 1], X[:, 2], c=y, s=20)
        elif dim==2:
            self.ax.scatter(X[:, 0],X[:, 1], c=y, s=20)
        elif dim==1:
            self.ax.scatter(X, np.zeros(len(X)), c=y, s=20)


    def updateUI(self, train_x, train_y, test_x, test_y, dataset_name, som, show_centroid_edges):
        self.fig.clear()

        if train_x is not None:
            dim = 3 if len(train_x[0]) >= 3 else len(train_x[0])
        elif test_x is not None:
            dim = 3 if len(test_x[0]) >= 3 else len(test_x[0])
        else:
            dim = 2


        if dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        self.ax.clear()

        self.plotDatasetInfo(dataset_name, dim)
        


        if train_x is not None:
            self.plotNodes(train_x, train_y, dim)

        if test_x is not None:
            self.plotNodes(test_x, test_y, dim)

        if som is not None:
            self.plotNetwork(som, dim, show_centroid_edges)

        self.draw()



class MyMainWindow(main_window_class):
    def __init__(self) -> None:
        super().__init__()

        self.main_ui = main_ui_class()
        self.main_ui.setupUi(self)

        # initialize
        self.dataset_manager = DatasetManager()
        self.som=None
        self.preds=None # predictions of test data

        self.setUpExtraWidgets()
        self.setUpUiEvent()

    def setUpExtraWidgets(self):
        self.canvas = MyDatasetCanvas()
        self.canvas.setParent(self.main_ui.figure_placeholder)

    def setUpUiEvent(self):
        self.main_ui.select_file_btn.clicked.connect(self.QselectFileDialog)
        self.main_ui.show_train_data_cbox.stateChanged.connect(self.updateUI)
        self.main_ui.show_test_data_cbox.stateChanged.connect(self.updateUI)
        self.main_ui.show_som_centroids_cbox.stateChanged.connect(self.updateUI)
        self.main_ui.show_error_pred_cbox.stateChanged.connect(self.updateUI)
        self.main_ui.show_som_centroids_edges_cbox.stateChanged.connect(self.updateUI)

        self.main_ui.normalized_cbox.stateChanged.connect(self.dataformatChange)
        self.main_ui.pca_dim_slider.valueChanged[int].connect(self.dataformatChange)

        self.main_ui.start_train_btn.clicked.connect(self.trainNetwork)
        self.main_ui.start_test_btn.clicked.connect(self.testNetwork)
        self.main_ui.pca_dim_slider.setValue(0)

    def QselectFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, '選擇資料庫檔案',
                                            os.getcwd(), "Text files (*.txt *.csv *.tsv)")
        if len(fname[0])==0: return

        filepath = Path(fname[0])
        if filepath.exists():
            self.loadDatasetAndInitSOM(filepath)
            self.main_ui.selected_file_label.setText(self.dataset_manager.dataset_name)
        else:
            print('file not exist')

    def loadDatasetAndInitSOM(self, dataset_path):
        assert dataset_path.exists(), "file not exists"

        # load data parameters
        split_data = True
        split_ratio = 0.33333

        # load data, and process them
        self.dataset_manager.load_dataset(dataset_path=dataset_path, split_train_test=split_data, split_ratio=split_ratio)
        self.dataformatChange()

    def dataformatChange(self):
        self.initializeNetwork()
        self.updateUI()

    def updateUI(self):
        # display parameters
        show_train_nodes = self.main_ui.show_train_data_cbox.isChecked()
        show_test_nodes =  self.main_ui.show_test_data_cbox.isChecked()
        show_error_nodes = self.main_ui.show_error_pred_cbox.isChecked()
        show_centroids = self.main_ui.show_som_centroids_cbox.isChecked()
        show_centroid_edges = self.main_ui.show_som_centroids_edges_cbox.isChecked()

        normalized = self.main_ui.normalized_cbox.isChecked()
        pca_num = self.main_ui.pca_dim_slider.value()
        self.main_ui.show_pca_dim_label.setText(f"{pca_num}")

        train_x, train_y, test_x, test_y = None, None, None, None
        if show_train_nodes and self.dataset_manager.has_train_data:
            train_x, train_y = self.dataset_manager.getXy(is_test=False, pca_num=pca_num, normalized=normalized)
        
        if self.dataset_manager.has_test_data:
            test_x, test_y = self.dataset_manager.getXy(is_test=True, pca_num=pca_num, normalized=normalized)

            if self.preds is not None:

                correct_color_list = []
                correct = 0
                for i in range(len(self.preds)):
                    if self.preds[i] == test_y[i]:
                        correct += 1
                        correct_color_list.append('black')
                    else:
                        correct_color_list.append("red")
                if show_error_nodes:
                    test_y = correct_color_list
                self.main_ui.show_test_accuracy_label.setText(f"{correct/len(self.preds):3f}")
                
            if not show_test_nodes:
                test_x, test_y  = None, None
        self.canvas.updateUI(train_x, train_y, test_x, test_y, self.dataset_manager.dataset_name, self.som if show_centroids else None , show_centroid_edges)



    def initializeNetwork(self):
        if not self.dataset_manager.has_data:
            # print('need to load dataset before initialize som network')
            return 
        # get proccessed data
        normalized = self.main_ui.normalized_cbox.isChecked()
        pca_num = self.main_ui.pca_dim_slider.value()

        train_x, _ = self.dataset_manager.getXy(is_test=False, pca_num=pca_num, normalized=normalized)

        # network paramaters
        _, input_dim = train_x.shape
        width, height = 15, 15
        kernel_mean = 0
        kernel_std = 2
        kernel_size = (5, 5)
        conscience_C = 0.9
        win_rate_decay = 1e-4
        lr_std_decay_constant = 150

        # fmt: off
        self.som = SOMnetwork(
            input_dim, width, height,
            kernel_mean, kernel_std, kernel_size,
            conscience_C, win_rate_decay, lr_std_decay_constant,
        )
        self.preds = None

        self.main_ui.show_test_accuracy_label.setText('Not trained yet')

    def trainNetwork(self):
        if self.som is None or not self.dataset_manager.has_train_data:
            return

        # train paramaters
        epochs = self.main_ui.epochs_spin_box.value()
        learning_rate = self.main_ui.learning_rate_spin_box.value()
        # print(f'train newwork with {epochs} epochs and {learning_rate:.2f} learning rate')

        normalized = self.main_ui.normalized_cbox.isChecked()
        pca_num = self.main_ui.pca_dim_slider.value()

        train_x, train_y = self.dataset_manager.getXy(is_test=False, pca_num=pca_num, normalized=normalized)


        self.som.train(train_x, epochs=epochs, learning_rate=learning_rate)
        self.som.mapCentroidsToClasses(train_x, train_y)

        self.updateUI()

    def testNetwork(self):
        if self.som is None or not self.dataset_manager.has_test_data:
            return

        # get proccessed data
        normalized = self.main_ui.normalized_cbox.isChecked()
        pca_num = self.main_ui.pca_dim_slider.value()

        test_x, _ = self.dataset_manager.getXy(is_test=True, pca_num=pca_num, normalized=normalized)

        self.preds = self.som.predict(test_x)

        self.updateUI()


