import numpy as np


class BayesianLR(object):
    """Bayesian Linear Regression.
    """
    def __init__(self, n_features, prior=None):
        self._n = n_features
        if prior is not None:
            xtx, xty = prior
            assert xtx.shape == (n_features, n_features)
            assert xty.shape == (n_features, 1)
            self._a = xtx
            self._b = xty
            self._beta = np.linalg.pinv(self._a) @ self._b
        else:
            self._a = np.zeros((n_features, n_features))
            self._b = np.zeros((n_features, 1))
            self._beta = np.zeros((n_features, 1))

    @property
    def num_features(self):
        return self._n

    @property
    def prior(self):
        return self._a, self._b

    @property
    def beta(self):
        return self._beta

    def bayeslr(self, xtx, xty, decay_factor=1):
        self._a = decay_factor * self._a + xtx
        self._b = decay_factor * self._b + xty
        self._beta = np.linalg.pinv(self._a) @ self._b
        return self.beta
