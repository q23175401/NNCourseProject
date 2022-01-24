import math
import numpy as np
import networkx as nx

class CompetitiveNetwork:
    def __init__(self, input_dim, n_units, conscience_C=10, win_rate_decay=1e-4, lr_std_decay_constant=100) -> None:
        
        self.n_units = n_units
        self.input_dim = input_dim
        self.conscience_C = conscience_C
        self.win_rate_decay = win_rate_decay
        self.lr_std_decay_constant = lr_std_decay_constant

        self._resetWeights()

    def _resetWeights(self):
        self.W = np.random.normal(0, 1, (self.n_units, self.input_dim))
        self.W_correspond_classes = np.arange(self.n_units) # if not map to y
        self.win_rates = np.ones((self.n_units))/self.n_units # set p = 1/N
        self.trained_epochs = 0


    def predict(self, X):
        closest_Wargs = self._forward(X, is_train=False)
        for i, w_arg in enumerate(closest_Wargs):
            # set w to correspond classes
            closest_Wargs[i] = self.W_correspond_classes[w_arg]

        return closest_Wargs

    def train(self, X, epochs=10, learning_rate=0.5):
        """
        Args:
            X: all data X with shape [batch_size, input_dim]
        """

        ### map all Ws onto X
        for ep in range(epochs):
            random_idxs = np.arange(len(X))
            np.random.shuffle(random_idxs)
            for ridx in random_idxs:
                x = X[ridx : ridx + 1]

                new_learning_rate = self._getNewLearningRate(learning_rate)
                self._updateW(x, new_learning_rate)

            # decay happends after each epoch
            self.trained_epochs += 1

    def mapCentroidsToClasses(self, train_x, train_y):
        ### after train W to X
        ### decide each w to be what class

        assert len(train_x) == len(train_y), 'data shape not matched'

        for wi in range(self.n_units):
            w = self.W[wi:wi+1]

            # find closest x to this w
            dists_squared = np.sum((w - train_x)**2, axis=1)
            most_close_x_arg = dists_squared.argmin()

            # get correspond class of this x
            correspond_class = train_y[most_close_x_arg]

            # set the class to this w
            self.W_correspond_classes[wi] = correspond_class

    def _forward(self, X, is_train=False):
        """
        Args:
            X: all data X with shape [batch, input_dim]
        """
        arg_list = []
        for xi in range((len(X))):
            x = X[xi : xi + 1]

            b = self.conscience_C * (1/self.n_units - self.win_rates) if is_train else 0 # conscience mechanism
            dists_squared = np.sum((x - self.W) ** 2, axis=1) - b
            maxSimilarityArgsort = dists_squared.argmin()
            self._updateWinRate(maxSimilarityArgsort)

            arg_list.append(maxSimilarityArgsort)
        return np.array(arg_list)

    def _updateW(self, x, learning_rate):
        """
        Args:
            x: one data x with shape [1, input_dim]
        """

        mSA = self._forward(x)[0] # maxSimilarityArg
        self.W[mSA : mSA + 1] += learning_rate * (x - self.W[mSA : mSA + 1])

    def _getNewLearningRate(self, learning_rate):
        new_learning_rate =  learning_rate * math.exp(-self.trained_epochs/self.lr_std_decay_constant)
        return new_learning_rate
    
    def _updateWinRate(self, winner_arg):
        # conscience mechanism
        for ri, wr in enumerate(self.win_rates):
            y = 1 if ri == winner_arg else 0
            self.win_rates[ri] += self.win_rate_decay * (y-wr)



class SOMnetwork(CompetitiveNetwork):
    def __init__(self, input_dim, width=10, height=10, kernel_mean=0, kernel_std=1, kernel_size=(5, 5), conscience_C=10, win_rate_decay=1e-4, lr_std_decay_constant=100) -> None:
        n_units = height * width
        super().__init__(input_dim, n_units, conscience_C, win_rate_decay, lr_std_decay_constant)

        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.kernel_mean = kernel_mean
        self.kernel_std = kernel_std
        # fmt: off
        self.neighbor_func = self._generate_neighbor_func(width, height, kernel_mean, kernel_std, kernel_size)
        # self.printNeighborFunc()

        self._setSOMGraph()

    def _setSOMGraph(self):
        num_centroids = self.n_units

        adjacency_matrix = np.zeros([num_centroids, num_centroids])

        for node_i in range(num_centroids):
            if node_i % self.width != self.width - 1:
                adjacency_matrix[node_i, node_i + 1] = 1
                adjacency_matrix[node_i + 1, node_i] = 1

            if node_i + self.height < num_centroids:
                adjacency_matrix[node_i, node_i + self.height] = 1
                adjacency_matrix[node_i + self.height, node_i] = 1
        self.G = nx.from_numpy_matrix(adjacency_matrix)

    def printNeighborFunc(self):
        '''
        just to see ratio of my neighbor function
        '''
        if self.width*self.height < self.kernel_size[0]*self.kernel_size[1]:
            print("your centroid number if less than kernel number")
            return

        s = ""
        for wi, ratio in self.neighbor_func(self.kernel_size[0]//2+self.width*(self.kernel_size[1]//2)):
            s += f"{ratio:.4f} " if (wi+1) % self.kernel_size[0]!=0 else f"{ratio:.4f}\n"
        print(s)

    def getCentroidsAndGraph(self):
        return self.W, self.G

    def _generate_neighbor_func(self, width, height, kernel_mean, kernel_std, kernel_size):
        # define a filter function
        def kernel(middle_idx):
            kw, kh = kernel_size
            k_mid_x, k_mid_y = kw // 2, kh // 2

            # recalculate ratio function
            start_idx = middle_idx - (k_mid_x + k_mid_y * width)
            for ky in range(kh):
                for kx in range(kw): # count row first
                    # id - middle => mean is 0
                    dist = math.sqrt((k_mid_x - kx) ** 2 + (k_mid_y - ky) ** 2)
                    new_std = self._getNewStd(kernel_std)
                    ratio_value = self._gaussian(dist, kernel_mean, new_std)

                    offset_idx = kx + ky * width
                    value_idx = start_idx + offset_idx
                    if 0 <= value_idx < height * width:
                        yield value_idx, ratio_value

        # return the neighbor function
        return kernel
 
    def _getNewStd(self, old_std):
        new_std = old_std * math.exp(-self.trained_epochs / self.lr_std_decay_constant)

        return new_std
        # return old_std

    def _gaussian(self, x, mean=0, std=1):
        return math.exp(-((x - mean) ** 2) / (2 * std ** 2 + 1e-10))

    def _updateW(self, x, learning_rate):
        """
        Args:
            x: one data x with shape [1, input_dim]
        """

        maxSimilarityArg = self._forward(x, is_train=True)[0]

        for ni, ratio in self.neighbor_func(maxSimilarityArg):
            self.W[ni : ni + 1] += learning_rate * ratio * (x - self.W[ni : ni + 1])

        

