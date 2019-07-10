import math
import numpy as np
import numpy.random as rng
import pandas as pd

__author__ = 'mcm'

# TODO ne pas scinder le dataset, seulement calculer les index et creer des vues sur un tableau central


class ModelFramework(object):
    def __init__(self):
        # initialize the datasets
        self.x_train = np.asarray([])
        self.y_train = np.asarray([])
        self.x_valid = np.asarray([])
        self.y_valid = np.asarray([])
        self.x_test = np.asarray([])
        self.y_test = np.asarray([])

        # initialize the meta parameters
        self.threshold = 0.5
        self.learning_factor = 0.1
        self.regularization_factor = 1.0

    def load_data(self, path_x, path_y, use='train', columns_to_drop=None):
        if path_x and path_y:
            # open the files
            print('loading {}...'.format(path_x))
            x_set = pd.read_csv(path_x)
            print('loading {}...'.format(path_y))
            y_set = pd.read_csv(path_y, header=None)

            # drop the columns specified by the user
            self._drop_columns(x_set, columns_to_drop)

            # get the values out of the pandas dataframe => numpy array
            x_set = x_set.values
            y_set = y_set.values
            y_set = y_set.reshape(y_set.shape[0])  # otherwise y is taken as a 2 dimensional array

            # put the data in the right place
            if use == 'train':
                self.x_train = x_set
                self.y_train = y_set
            elif use == 'validation':
                self.x_valid = x_set
                self.y_valid = y_set
            elif use == 'test':
                self.x_test = x_set
                self.y_test = y_set

            # do a random permutation on the sets
            self.shuffle_data()

    def split_data(self, split_percent=(0.6, 0.4)):
        split_indices = []
        acc = 0.
        size = len(self.x_train)  # by default we split the training set

        # calculate the split positions in the training set
        if sum(split_percent) == 1.0:
            for fraction in split_percent:
                acc += fraction
                split_indices.append(int(math.floor(acc * size)))
        else:
            split_indices.append(int(math.floor(0.6 * size)))
            split_indices.append(size)

        split_indices.pop()

        # actually split the training set
        x_split = list(np.split(self.x_train, split_indices))
        y_split = list(np.split(self.y_train, split_indices))
        if len(split_indices) == 1:
            self.x_train = x_split[0]
            self.x_valid = x_split[1]
            self.y_train = y_split[0]
            self.y_valid = y_split[1]
        elif len(split_indices) == 2:
            self.x_train = x_split[0]
            self.x_valid = x_split[1]
            self.x_test = x_split[2]
            self.y_train = y_split[0]
            self.y_valid = y_split[1]
            self.y_test = y_split[2]
        pass

    def shuffle_data(self):
        train_size = len(self.x_train)
        valid_size = len(self.x_valid)
        test_size = len(self.x_test)

        # do a random permutation on x and y
        print('shuffling the training set...')
        permutation_index = rng.permutation(train_size)
        self.x_train = self.x_train[permutation_index]
        self.y_train = self.y_train[permutation_index]
        print('shuffling the validation set...')
        permutation_index = rng.permutation(valid_size)
        self.x_valid = self.x_valid[permutation_index]
        self.y_valid = self.y_valid[permutation_index]
        print('shuffling the testing set...')
        permutation_index = rng.permutation(test_size)
        self.x_test = self.x_test[permutation_index]
        self.y_test = self.y_test[permutation_index]

    def normalize_data(self):
        pass

    def set_batch(self, batch_size):
        pass

    def set_computation(self, learning_factor, regularization_factor, max_epochs):
        pass

    def save_model(self, path):
        pass

    def train(self):
        pass

    def _drop_columns(self, df, columns):
        if df is not None:
            for col in columns:
                print('dropping {}...'.format(col))
                df.drop(col, axis=1, inplace=True)
