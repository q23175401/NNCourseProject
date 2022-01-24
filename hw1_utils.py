# my modules
from NN.loss import CategoricalCrossEntropy
from NN.network import Network
from NN.layer import DenseLayer, Tanh, Softmax, ReLU

# libraries
from pathlib import Path
import numpy as np
import sys
import os
from NN.optimizer import SGD, Adam
from pca import PCA


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def OneHotEncode(y):
    label_arg_dict = {}
    label_id = -1
    # get all labels in y
    for label in y:
        if label not in label_arg_dict.keys():
            label_id += 1
            label_arg_dict[label] = label_id

    # get corresponding encode value from dictionary
    onehot_y = np.zeros([len(y), len(label_arg_dict.items())])
    for li in range(len(y)):
        onehot_y[li, label_arg_dict[y[li]]] = 1

    return onehot_y, label_arg_dict


def load_dataset(filename, normalized=False):

    with open(filename, "r", encoding="utf-8") as f:
        datalines = f.readlines()

    for di in range(len(datalines)):

        try:
            datalines[di] = datalines[di].split(" ")
            datalines[di] = [float(d) for d in datalines[di]]
        except Exception as e:
            print(e)
            return np.array([]), np.array([])

    datalines = np.array(datalines, dtype=np.float64)
    assert len(datalines) > 0, "no data line is in the file."

    x = datalines[:, :-1]
    y = datalines[:, -1]

    if normalized:
        x = (x - x.mean()) / (x.std() + 1e-10)

    return x, y


def get_train_test_data(X, y, ratio=0.1):
    testing_len = int(len(X) * ratio)

    random_select = np.arange(len(X))
    np.random.shuffle(random_select)
    testing_select = random_select[:testing_len]
    training_select = random_select[testing_len:]

    test_x, test_y = X[testing_select], y[testing_select]
    train_x, train_y = X[training_select], y[training_select]
    return train_x, test_x, train_y, test_y


def train_dataset(dataset_name, learning_rate, batch_size, epochs, normalized=True):

    X, y = load_dataset(dataset_name, normalized)
    one_hot_y, label_arg_dict = OneHotEncode(y)

    if len(y) != 4 and len(y) != 8:

        train_x, test_x, train_one_hot_y, test_one_hot_y = get_train_test_data(X, one_hot_y, 0.3333)
    else:
        train_x, test_x, train_one_hot_y, test_one_hot_y = X, X, one_hot_y, one_hot_y  # 因為只有4、8筆資料

    MyNet = Network(
        [
            DenseLayer(len(X[0]), 64),
            Tanh(),
            DenseLayer(64, 32),
            Tanh(),
            DenseLayer(32, 16),
            Tanh(),
            DenseLayer(16, len(train_one_hot_y[0])),
            # Sigmoid(),
            Softmax(),
            # Tanh(),
        ],
        learning_rate,  # specify learning rate of sgd
        CategoricalCrossEntropy(),
        # LeastSquaredError(),
    )

    # train model
    MyNet.Train(train_x, train_one_hot_y, batchsize=batch_size, epochs=epochs, verbose=0)

    # test model
    prediction = MyNet.Predict(test_x).argmax(axis=1)
    correct = 0
    for p, onehot_ans in zip(prediction, test_one_hot_y):
        if p == onehot_ans.argmax():
            correct += 1
    print("dataset: ", dataset_name, " accuracy: ", correct / len(test_x) if len(test_x) != 0 else "NoTestData")


def printAccuracy(predictions, answers):
    correct = 0
    for p, a in zip(predictions, answers):
        if p == a:
            correct += 1
    print(
        "dataset: ",
        dataset_path.name,
        " test accuracy: ",
        correct / len(predictions) if len(predictions) != 0 else "NoTestData",
    )


def get2DInfos(x_datas, perceptronLayer):
    def F2D(x1, i):
        # y = (w[0, i]x + b) /w[1, i]
        return (perceptronLayer.W[0, i] * x1 + perceptronLayer.B[0, i]) / (-perceptronLayer.W[1, i] + 1e-10)

    x2_min = x_datas[0, 1]
    x2_max = x_datas[0, 1]

    x1_min = x_datas[0, 0]
    x1_max = x_datas[0, 0]

    for xd in x_datas:
        x1_cur = xd[0]
        x2_cur = xd[1]

        x1_min = x1_cur if x1_cur < x1_min else x1_min
        x1_max = x1_cur if x1_cur > x1_max else x1_max

        x2_min = x2_cur if x2_cur < x2_min else x2_min
        x2_max = x2_cur if x2_cur > x2_max else x2_max

    return x1_min, x1_max, x2_min, x2_max, F2D


def get3DInfos(x_datas, perceptronLayer):
    def F3D(x1, x2, i):
        # y = (w[0, i]x + w[1, i]x2 + b[0, i]) /w[2, i]
        return (perceptronLayer.W[0, i] * x1 + perceptronLayer.W[1, i] * x2 + perceptronLayer.B[0, i]) / (
            -perceptronLayer.W[2, i] + 1e-10
        )

    x3_min = x_datas[0, 2]
    x3_max = x_datas[0, 2]

    x2_min = x_datas[0, 1]
    x2_max = x_datas[0, 1]

    x1_min = x_datas[0, 0]
    x1_max = x_datas[0, 0]

    for xd in x_datas:
        x1_cur = xd[0]
        x2_cur = xd[1]
        x3_cur = xd[2]

        x1_min = x1_cur if x1_cur < x1_min else x1_min
        x1_max = x1_cur if x1_cur > x1_max else x1_max

        x2_min = x2_cur if x2_cur < x2_min else x2_min
        x2_max = x2_cur if x2_cur > x2_max else x2_max

        x3_min = x3_cur if x3_cur < x3_min else x3_min
        x3_max = x3_cur if x3_cur > x3_max else x3_max

    return x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, F3D


class DatasetManager:
    def __init__(self):
        self.dataset_path = None
        self.train_mean = None
        self.train_std = None
        self.X, self.y = None, None  # the whole dataset
        self.train_x, self.train_y = None, None  # training part
        self.test_x, self.test_y = None, None  # testing part
        self.pca = None
        self.pca_norm = None

    @property
    def dataset_name(self):
        return self.dataset_path.name if self.dataset_path is not None else ""

    def getXy(self, is_test=False, pca_num=0, normalized=False, onehot=False):
        if not is_test:
            x = self.train_x
            y = self.train_y
        else:
            x = self.test_x
            y = self.test_y

        assert x is not None and y is not None, "the data you want to load is not loaded yet"
        # use pca to map Xdata to 3 dim

        return_y = self._oneHotEncodeUsingDict(y, self.label_arg_dict) if onehot else y

        return_x = (x - self.train_mean) / (self.train_std + 1e-10) if normalized else x

        if pca_num > 0 and len(return_x) > 0 and pca_num <= len(return_x[0]):
            if normalized:
                return_x = self.pca_norm.transform(return_x, pca_num)
            else:
                return_x = self.pca.transform(return_x, pca_num)

        return return_x, return_y

    def _getFitPCA(self, train_x, normalized=False):
        if normalized:
            self.pca_norm = PCA()
            pca = self.pca_norm
        else:
            self.pca = PCA()
            pca = self.pca
        pca.fit(train_x)

    def load_dataset(self, dataset_path: Path, is_test_data=False, split_train_test=False, split_ratio=0.1):
        """
        dataset_path Path:
            the Path of the dataset you want to load

        is_test_data bool:
            if the data is for test, test_x and test_y will be loaded

        split_train_test bool:
            if the data is the whole dataset and you want to split into train and test,
            X, y, test_x, and test_y will be loaded

        split_ratio bool:
            test data ratio when you want to split train and test data
        """
        if not dataset_path.exists():
            print(dataset_path.name + " is not exsits")
            return

        self.dataset_path = dataset_path
        with open(str(dataset_path.absolute()), "r", encoding="utf-8") as f:
            datalines = f.readlines()

        for di in range(len(datalines)):

            try:
                datalines[di] = datalines[di].split(" ")
                datalines[di] = [float(d) for d in datalines[di]]
            except Exception as e:
                print(e)
                return

        datalines = np.array(datalines, dtype=np.float64)
        assert len(datalines) > 0, "No data line is in the file."

        x = datalines[:, :-1]
        y = datalines[:, -1]
        self.X = x
        self.y = y

        if is_test_data:
            self.test_x = x
            self.test_y = y

            assert self.train_x, "You have not loaded train_data"
            assert len(self.train_x[0]) != len(self.test_x[0]), "Test data dim does not match training data"

        elif split_train_test:
            train_x, test_x, train_y, test_y = self._split_train_test_data(x, y, split_ratio)

            self.train_mean = train_x.mean()
            self.train_std = train_x.std()
            self._getOneHotEncodeDict(train_y)

            self.train_x = train_x
            self._getFitPCA(self.train_x)
            self._getFitPCA(self.train_x, normalized=True)

            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y
        else:
            self.train_mean = x.mean()
            self.train_std = x.std()
            self._getOneHotEncodeDict(y)

            self.train_x = x
            self._getFitPCA(self.train_x)
            self._getFitPCA(self.train_x, normalized=True)

            self.train_y = y

    def _split_train_test_data(self, X, y, test_ratio=0.1):
        testing_len = int(len(X) * test_ratio)

        random_select = np.arange(len(X))
        np.random.shuffle(random_select)
        testing_select = random_select[:testing_len]
        training_select = random_select[testing_len:]

        test_x, test_y = X[testing_select], y[testing_select]
        train_x, train_y = X[training_select], y[training_select]
        return train_x, test_x, train_y, test_y

    def _getOneHotEncodeDict(self, y):
        label_arg_dict = {}
        label_id = -1
        # get all labels in y
        for label in y:
            if label not in label_arg_dict.keys():
                label_id += 1
                label_arg_dict[label] = label_id
        self.label_arg_dict = label_arg_dict

    def _oneHotEncodeUsingDict(self, y, label_arg_dict):
        # get corresponding encode value from dictionary
        onehot_y = np.zeros([len(y), len(label_arg_dict.items()) + 1])
        for li in range(len(y)):
            onehot_y[li, label_arg_dict.get(y[li], -1)] = 1

        return onehot_y


if __name__ == "__main__":
    import glob

    datasetname = "./NN_HW1_Dataset/*.txt"
    filenames = glob.glob(datasetname)

    learning_rate = 1e-3
    batch_size = 16
    epochs = 1000
    dataset_path = Path(filenames[10])
    print(dataset_path)

    dm = DatasetManager()
    dm.load_dataset(dataset_path, is_test_data=False, split_train_test=True, split_ratio=0.333)

    train_x, train_y = dm.getXy(is_test=False, normalized=True, onehot=True, pca_num=3)
    test_x, test_y = dm.getXy(is_test=True, normalized=True, onehot=True, pca_num=3)

    layer1_line = 4
    layer2_line = 32
    dataset_dim = len(train_x[0])
    MyNet = Network(
        [
            DenseLayer(dataset_dim, layer1_line),
            # Tanh(),
            ReLU(),
            # DenseLayer(layer1_line, layer2_line),
            # ReLU(),
            # Tanh(),
            DenseLayer(layer1_line, len(train_y[0])),
            Softmax(),
        ],
        Adam(learning_rate),  # specify learning rate of sgd
        CategoricalCrossEntropy(),
    )

    def prediction_func(y: np.ndarray):
        return y.argmax(axis=1)

    # train model
    MyNet.Train(
        train_x, train_y, batchsize=batch_size, epochs=epochs, verbose=2, select_prediction_func=prediction_func
    )

    # test model
    predictions = MyNet.Predict(test_x, prediction_func)
    answers = prediction_func(test_y)
    printAccuracy(predictions, answers)

    # plot line
    import matplotlib.pyplot as plt

    if dataset_dim == 2:
        plt.scatter(test_x[:, 0], test_x[:, 1], c=test_y.argmax(axis=1), s=10)

        perceptronLayer = MyNet.layerStack[0]
        x1_min, x1_max, x2_min, x2_max, F2D = get2DInfos(test_x, perceptronLayer)

        plt.ylim(x2_min, x2_max)
        plt.title(dataset_path.name)
        x1_line = np.linspace(x1_min, x1_max, 100)

        for i in range(layer1_line):
            x2_value = F2D(x1_line, i)
            plt.plot(x1_line, x2_value)

        plt.show()
    elif dataset_dim >= 3:

        perceptronLayer = MyNet.layerStack[0]

        x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, F3D = get3DInfos(test_x, perceptronLayer)

        ax = plt.axes(projection="3d")
        ax.scatter3D(test_x[:, 0], test_x[:, 1], test_x[:, 2], c=test_y.argmax(axis=1), s=10)

        x1_line = np.linspace(x1_min, x1_max, 20)
        x2_line = np.linspace(x2_min, x2_max, 20)
        x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)
        for i in range(layer1_line):
            x3_value = F3D(x1_mesh, x2_mesh, i)
            # ax.plot_surface(x1_mesh, x2_mesh, z_mesh, color='red')
            ax.plot_wireframe(x1_mesh, x2_mesh, x3_value, color=(0.3 * i, 0.5, 0.3))

        ax.set_zlim(x3_min, x3_max)
        plt.show()
