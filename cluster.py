import os
import time
import math
import warnings
import numpy as np
import scipy.sparse as sparse

from sklearn.cluster import KMeans
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import sparse_encode
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClusterMixin


class SelfRepresentation(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, affinity='symmetrize', random_state=None, n_init=20, n_jobs=1):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.random_state = random_state
        self.n_init = n_init
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64)
        time_base = time.time()
        self._self_representation(X)
        self.timer_self_representation_ = time.time() - time_base
        self._representation_to_affinity()
        self._spectral_clustering()
        self.timer_time_ = time.time() - time_base
        return self

    def fit_self_representation(self, X, y=None):
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64)
        time_base = time.time()
        self._self_representation(X)
        self.timer_self_representation_ = time.time() - time_base
        return self

    def _representation_to_affinity(self):
        Z = normalize(self.representation_matrix_, norm='l2')
        if self.affinity == 'symmetrize':
            self.affinity_matrix_ = 0.5 * (np.abs(Z) + np.abs(Z.T))
        elif self.affinity == 'nearest_neighbors':
            neighbors_graph = kneighbors_graph(Z, 3, mode='connectivity', include_self=False)
            self.affinity_matrix_ = 0.5 * (neighbors_graph + neighbors_graph.T)

    def _spectral_clustering(self):
        from sklearn.utils import check_symmetric
        from scipy.sparse.csgraph import laplacian
        from scipy.sparse.linalg import eigsh

        A = check_symmetric(self.affinity_matrix_)
        L = laplacian(A, normed=True)
        _, vec = eigsh(sparse.identity(L.shape[0]) - L, k=self.n_clusters, which='LA')
        embedding = normalize(vec)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        self.labels_ = kmeans.fit_predict(embedding)


class ElasticNetSubspaceClustering(SelfRepresentation):
    def __init__(self, n_clusters=8, affinity='symmetrize', random_state=None, n_init=20, n_jobs=1,
                 gamma=50.0, gamma_nz=True, tau=1.0, algorithm='lasso_lars',
                 active_support=False, active_support_params=None, n_nonzero=50):
        super().__init__(n_clusters, affinity, random_state, n_init, n_jobs)
        self.gamma = gamma
        self.gamma_nz = gamma_nz
        self.tau = tau
        self.algorithm = algorithm
        self.active_support = active_support
        self.active_support_params = active_support_params or {}
        self.n_nonzero = n_nonzero

    def _self_representation(self, X):
        self.representation_matrix_ = elastic_net_subspace_clustering(
            X, gamma=self.gamma, gamma_nz=self.gamma_nz, tau=self.tau,
            algorithm=self.algorithm, active_support=self.active_support,
            active_support_params=self.active_support_params, n_nonzero=self.n_nonzero)


def elastic_net_subspace_clustering(X, gamma=50.0, gamma_nz=True, tau=1.0,
                                    algorithm='lasso_lars', active_support=False,
                                    active_support_params=None, n_nonzero=50):
    if algorithm in ('lasso_lars', 'lasso_cd') and tau < 1.0 - 1e-10:
        warnings.warn(f"algorithm {algorithm} cannot handle tau < 1. Forcing tau=1.0")
        tau = 1.0

    n_samples = X.shape[0]
    rows, cols, vals = [], [], []

    for i in tqdm(range(n_samples), desc="EnSC"):
        y = X[i, :].copy().reshape(1, -1)
        X[i, :] = 0

        if gamma_nz:
            coh = np.abs(X @ y.T)
            alpha0 = coh.max() / tau
            alpha = alpha0 / gamma
        else:
            alpha = 1.0 / gamma

        c = sparse_encode(y, X, algorithm=algorithm, alpha=alpha)[0]

        nonzero_idx = np.flatnonzero(c)
        if nonzero_idx.size > n_nonzero:
            top_idx = np.argsort(-np.abs(c[nonzero_idx]))[:n_nonzero]
            nonzero_idx = nonzero_idx[top_idx]

        rows.extend([i] * len(nonzero_idx))
        cols.extend(nonzero_idx.tolist())
        vals.extend(c[nonzero_idx].tolist())

        X[i, :] = y

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))


def clustering_accuracy(true_labels, pred_labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / len(true_labels)
