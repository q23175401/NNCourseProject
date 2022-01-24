from hw3_som_utils import DatasetManager
from NN import SOMnetwork
from pathlib import Path
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication
from typing import List
from hw3_som_ui_controller import MyMainWindow
import sys


def plot_nodes(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20)

def plot_classification_results(som, test_x, test_y, show_error_nodes=False):
    """plot classification results"""

    # get cluster predictions x using correspond w->class mapping
    preds = som.predict(test_x)
    correct_color_list = []
    correct = 0
    for i in range(len(preds)):
        if preds[i] == test_y[i]:
            correct += 1
            correct_color_list.append('black')
        else:
            correct_color_list.append("red")

    print(f"Accuracy: {correct/len(preds):.4f}")
    # plot classification results
    plt.scatter(test_x[:, 0], test_x[:, 1], c=preds if not show_error_nodes else correct_color_list, s=20)

def plot_centroids_graph(som, width, height, show_centroid_edges=True):
    num_centroids = width * height

    """ plot nodes and edges using networkx module """
    # adjacency_matrix = np.zeros([num_centroids, num_centroids])

    # for node_i in range(num_centroids):
    #     if node_i % width != width - 1:
    #         adjacency_matrix[node_i, node_i + 1] = 1
    #         adjacency_matrix[node_i + 1, node_i] = 1

    #     if node_i + height < num_centroids:
    #         adjacency_matrix[node_i, node_i + height] = 1
    #         adjacency_matrix[node_i + height, node_i] = 1
    # G = nx.from_numpy_matrix(adjacency_matrix)

    # # pos_dict = {ni: (ni % width, ni // height) for ni in range(len(som.W))} # square of nodes to see node connections
    # pos_dict = {ni: (som.W[ni, 0], som.W[ni, 1]) for ni in range(len(som.W))}
    # nx.draw(
    #     G,
    #     pos=pos_dict,
    #     node_size=20,
    #     edgelist=G.edges() if show_centroid_edges else [],
    # )

    # w = np.concatenate([np.array([ni % width for ni in range(num_centroids)]).reshape(1, num_centroids).T, np.array([ni // height for ni in range(num_centroids)]).reshape(1, num_centroids).T], axis=1)
    """ plot nodes and edges without networkx module """
    w = som.W
    plt.scatter(w[:, 0], w[:, 1], c='blue', s=20)
    if show_centroid_edges:
        for wi in range(0, num_centroids ,width):
            pos_list = []
            for d in range(2):
                pos_list.append(w[wi:wi+width, d])

            plt.plot(*pos_list, c='purple')
            # plt.plot(w[wi:wi+width, 0], w[wi:wi+width, 1], c='purple')
        
        for hi in range(0, width):
            pos_list = []
            for d in range(2):
                pos_list.append(w[hi:num_centroids:width, d])

            plt.plot(*pos_list, c='purple')



        # plt.plot(w[hi:num_centroids:width, 0], w[hi:num_centroids:width, 1], c='purple')

def test_som():
    dataset_path = Path("./NN_HW3_SOM_Dataset/2Circle1.TXT")

    assert dataset_path.exists, "file not exists"

    # load data, and process them
    dm = DatasetManager()
    dm.load_dataset(dataset_path=dataset_path, split_train_test=True, split_ratio=0.33)

    # get proccessed data
    train_x, train_y = dm.getXy(is_test=False, normalized=True)
    test_x, test_y = dm.getXy(is_test=True, normalized=True)

    # network paramaters
    _, input_dim = train_x.shape
    width, height = 15, 15
    kernel_mean = 0
    kernel_std = 2
    kernel_size = (5, 5)
    conscience_C = 0.9
    win_rate_decay = 1e-4
    lr_std_decay_constant = 150

    # train paramaters
    epochs = 100
    learning_rate = 0.7

    # display parameters
    show_train_nodes = True
    show_test_nodes = True
    show_error_nodes = True
    show_centroids = True
    show_centroid_edges = True

    # define network
    # fmt: off
    som = SOMnetwork(
        input_dim, width, height,
        kernel_mean, kernel_std, kernel_size,
        conscience_C, win_rate_decay, lr_std_decay_constant,
    )
    # show kernel ratios with current kernel_std
    # som.printNeighborFunc()

    # train centroids to map onto X and map centroids to classes of each x
    som.train(train_x, epochs, learning_rate)
    som.mapCentroidsToClasses(train_x, train_y)

    """ plot training nodes"""
    if show_train_nodes:
        plot_nodes(train_x, train_y)

    """ plot classification results """
    # get cluster predictions x using correspond w->class mapping
    if show_test_nodes:
        plot_classification_results(som, test_x, test_y, show_error_nodes)

    """ plot centroids graph """
    if show_centroids:
        plot_centroids_graph(som, width, height, show_centroid_edges=show_centroid_edges)

    plt.show()

class MyHw3App(QApplication):
    def __init__(self, argv: List[str]) -> None:
        super().__init__(argv)

        # create ui and window obj after app created
        self.main_window = MyMainWindow()

    def start(self):
        self.main_window.show()
        self.exec()


def main():
    myApp = MyHw3App(sys.argv)
    myApp.start()

if __name__ == "__main__":
    main()

    # test_som()