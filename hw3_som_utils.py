import numpy as np
from pca import PCA
from pathlib import Path
import os, sys


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class DatasetManager:
    def __init__(self):
        self.reset_data_manager()

    def reset_data_manager(self):
        self.dataset_path = None
        self.train_mean = None
        self.train_std = None
        self.X, self.y = None, None  # the whole dataset
        self.train_x, self.train_y = None, None  # training part
        self.test_x, self.test_y = None, None  # testing part
        self.pca = None
        self.pca_norm = None

        self.has_data = False
        self.has_train_data = False
        self.has_test_data = False

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

        return_y = self._oneHotEncodeUsingDict(y, self.label_arg_dict) if onehot else y

        return_x = (x - self.train_mean) / (self.train_std + 1e-10) if normalized else x

        # use pca to map Xdata to 3 dim
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
        load data and process it, use getXy to get specific format of data

        Args:
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

        # clear before reload
        self.reset_data_manager()

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
        self.has_data = True

        if is_test_data:
            assert self.train_x, "You have not loaded train_data"
            assert len(self.train_x[0]) != len(self.test_x[0]), "Test data dim does not match training data"

            self.test_x = x
            self.test_y = y

            self.has_test_data = True

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

            self.has_train_data = True
            self.has_test_data = True
        else:
            self.train_mean = x.mean()
            self.train_std = x.std()
            self._getOneHotEncodeDict(y)

            self.train_x = x
            self._getFitPCA(self.train_x)
            self._getFitPCA(self.train_x, normalized=True)

            self.train_y = y

            self.has_train_data = True

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
