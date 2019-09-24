"""
Defines various utility functions for use with DenserFlow.
"""
from typing import Tuple

import h5py
import numpy as np
from nptyping import Array

from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh


def h5_to_np(filename: str, key: str) -> Array[float]:
    """
    load a h5 file and return a dataset within it as a numpy array.
    :param filename: name of the h5 file to open
    :param key: key to lookup the dataset with inside the h5 file.
    """
    with h5py.File(filename, "r") as f:
        return np.copy(f[key])


def np_to_h5(filename: str, key: str, data: Array[float]) -> None:
    """
    Save a numpy array to a h5 file.
    :param filename: name of the h5 file to write to
    :param key: key to store the dataset within.
    """
    with h5py.File(filename, "w") as f:
        f.create_dataset(key, data=data)


def label_to_one_hot(labels: Array[int], num_classes: int) -> Array[float]:
    """
    Turn an array of label values into the equivalent set of one-hot encoded vectors.
    :param labels: array of labels
    :param num_classes: number of classes within the dataset
    """
    one_hots = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    # fun trick: arange gives us 0....num. samples, so this does nicely
    one_hots[np.arange(labels.shape[0]), labels] = 1
    return one_hots


def validate_split(
    x: Array[float], y: Array[float], split: float = 0.1
) -> Tuple[Array[float], Array[float], Array[float], Array[float]]:
    """
    Split a dataset into train and validation sets.
    returns x_train, y_train, x_test, y_test
    :param x: the array of inputs
    :param y: the array of target values
    :param split: the amount to split the dataset by
    """
    shuffle_indices = np.random.permutation(len(x))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = len(x) - int(len(x) * split)
    return x[:train_len], y[:train_len], x[train_len:], y[train_len:]


def StandardScalar(X):
    """
    Performs z scaling on the data (columnwise)
    :param: X: input data
    """
    return (X - np.mean(X)) / (np.std(X) + 1e-8)


def MinMaxScalar(X):
    """
    Performs minmax scaling scaling on the data (columnwise)
    :param X: input data
    """
    return (X - np.min(X)) / (np.ptp(X) + 1e-8)


def PCA_fit(X, n_components):

    """
    Computes the projection matrix for PCA_fit
    :param X: normalized training data
    :param n_components: the dimension of the reduced data
    Code based off:
    https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html

    """
    cov_mat = np.cov(X.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    num_features = X.shape[1]
    proj_mat = eig_pairs[0][1].reshape(num_features, 1)
    for eig_vec_idx in range(1, n_components):
        proj_mat = np.hstack(
            (proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features, 1))
        )

    return proj_mat


def PCA_predict(X, proj_mat):
    """
    Projects data onto dimension specified by the projection_matrix
    :param: X: normalized input data (can be train or test)
    :param: proj_mat: Projection matrix computed in PCA_fit
    """
    return X.dot(proj_mat)


def KernelPCA_fit(X, gamma, n_components):
    """
    Computes the projection eigen vectors and eigen values for PCA
        using the RBF kernel
    :param X: normalized training data
    :param gamma: kernel coefficient.
        The higher the gamma, the more 'influence' far datapoints have
    :param n_components: the dimension of the reduced data
    Code based off:
    https://sebastianraschka.com/Articles/2014_kernel_pca.html

    """
    # pairwise euclidian dist
    eu_dists = pdist(X, "sqeuclidean")

    # symmetric matrix
    sym_eu_dists = squareform(eu_dists)

    K = np.exp(-gamma * sym_eu_dists)

    # Centering
    N = K.shape[0]
    ones = np.ones((N, N)) / N
    K = K - ones.dot(K) - K.dot(ones) + ones.dot(K).dot(ones)

    eig_vals, eig_vecs = eigh(K)

    # getting eigenvals as matrix
    n_eig_vecs = np.column_stack((eig_vecs[:, -i] for i in range(1, n_components + 1)))
    n_eig_vals = [eig_vals[-i] for i in range(1, n_components + 1)]

    return n_eig_vecs, n_eig_vals


def KernelPCA_predict(X_new, X, gamma, n_eig_vecs, n_eig_vals):
    """
    Projects data onto dimension specified by the projection_matrix
    :param X_new: test data
    :param X: training data
    :param gamma: kernel parameter
    :param: n_eig_vecs: eigen vectors computed in KernelPCA_fit
    :param: n_eig_vals: eigen vectors  computed in KernelPCA_fit
    """
    # compute distance
    pair_dist = np.array([np.sum((X_new - row) ** 2) for row in X])
    # construct kernel matrix
    k = np.exp(-gamma * pair_dist)
    # project back
    return k.dot(n_eig_vecs / n_eig_vals)
