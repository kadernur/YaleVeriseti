import numpy as np
from scipy import linalg

from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin


class LocalityPreservingProjection(BaseEstimator, TransformerMixin):
   
    def __init__(self, n_components=2, n_neighbors=5,
                 weight='adjacency', weight_width=1.0,
                 neighbors_algorithm='auto'):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.weight = weight
        self.weight_width = weight_width
        self.neighbors_algorithm = neighbors_algorithm

    def fit(self, X, y=None):
        X = check_array(X)
        W = self._compute_weights(X)
        self.projection_ = self._compute_projection(X, W)
        return self

    def transform(self, X):
        X = check_array(X)
        return np.dot(X, self.projection_)

    def _compute_projection(self, X, W):
        X = check_array(X)

        D = np.diag(W.sum(1))
        L = D - W
        evals, evecs = eigh_robust(np.dot(X.T, np.dot(L, X)),
                                   np.dot(X.T, np.dot(D, X)),
                                   eigvals=(0, self.n_components - 1))
        return evecs

    def _compute_weights(self, X):
        X = check_array(X)
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm)
        self.nbrs_.fit(X)

        if self.weight == 'adjacency':
            W = kneighbors_graph(self.nbrs_, self.n_neighbors,
                                 mode='connectivity', include_self=True)
        elif self.weight == 'heat':
            W = kneighbors_graph(self.nbrs_, self.n_neighbors,
                                 mode='distance', include_self=True)
            W.data = np.exp(-W.data ** 2 / self.weight_width ** 2)
        else:
            raise ValueError("Unrecognized Weight")
        W = W.toarray()
        W = np.maximum(W, W.T)
        return W


def eigh_robust(a, b=None, eigvals=None, eigvals_only=False,
                overwrite_a=False, overwrite_b=False,
                turbo=True, check_finite=True):
    kwargs = dict(eigvals=eigvals, eigvals_only=eigvals_only,
                  turbo=turbo, check_finite=check_finite,
                  overwrite_a=overwrite_a, overwrite_b=overwrite_b)

    if b is None:
        return linalg.eigh(a, **kwargs)

    kwargs_b = dict(turbo=turbo, check_finite=check_finite,
                    overwrite_a=overwrite_b)  # b is a for this operation
    S, U = linalg.eigh(b, **kwargs_b)

    S[S <= 0] = np.inf
    Sinv = 1. / np.sqrt(S)
    W = Sinv[:, None] * np.dot(U.T, np.dot(a, U)) * Sinv
    output = linalg.eigh(W, **kwargs)

    if eigvals_only:
        return output
    else:
        evals, evecs = output
        return evals, np.dot(U, Sinv[:, None] * evecs)


