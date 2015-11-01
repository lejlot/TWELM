"""
Implementation assumes that there are TWO LABELS, namely -1 and +1.
If you have different labels you have to preproess them. Furthermore
be sure to correctly set "balanced" hyperparameter accordingly to the
metric you want to optimize
"""

import numpy as np
from scipy import linalg as la

def tanimoto(X, W, b=None):
    """ Tanimoto similarity function """
    XW = np.dot(X, W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return XW / (XX+WW-XW)    

class ELM(object):
    """ Extreme Learning Machine """

    def __init__(self, h, C=10000, f=tanimoto, random_state=666, balanced=False):
        """
        h - number of hidden units
        C - regularization strength (L2 norm)
        f - activation function [default: tanimoto]
        balanced - if set to true, model with maximize GMean (or Balanced accuracy), 
                   if set to false [default] - model will maximize Accuracy
        """
        self.h = h
        self.C = C
        self.f = f
        self.rs = random_state
        self.balanced = balanced

    def _hidden_init(self, X, y):
        """ Initializes hidden layer """
        np.random.seed(self.rs)
        W = csr_matrix(np.random.rand(self.h, X.shape[1]))
        b = np.random.normal(size=self.h)
        return W, b

    def fit(self, X, y):
        """ Fits ELM to training samples X and labels y """
        self.W, self.b = self._hidden_init(X, y)
        H = self.f(X, self.W, self.b)

        if self.balanced:
            counts = { l : float(y.tolist().count(l)) for l in set(y) }
            ms = max([ counts[k] for k in counts ])        
            self.counts = { l : np.sqrt( ms/counts[l] ) for l in counts }
        else:
            self.counts = { l: 1 for l in set(y) }

        w = np.array( [[ self.counts[a] for a in y ]] ).T
        H = np.multiply(H, w)
        y = np.multiply(y.reshape(-1,1), w).ravel()

        self.beta = la.inv(H.T.dot(H) + 1.0 / self.C * np.eye(H.shape[1])).dot((H.T.dot(y)).T)

    def predict(self, X):
        H = self.f(X, self.W, self.b)
        return np.array(np.sign(H.dot(self.beta)).tolist())


class XELM(ELM):
    """ Extreme Learning Machine initialized with training samples """

    def _hidden_init(self, X, y):

        h = min(self.h, X.shape[0]) # hidden neurons count can't exceed training set size

        np.random.seed(self.rs)
        W = X[np.random.choice(range(X.shape[0]), size=h, replace=False)]
        b = np.random.normal(size=h)
        return W, b


class TWELM(XELM):
    """ 
    TWELM* model from

    "Weighted Tanimoto Extreme Learning Machine with case study of Drug Discovery"
    WM Czarnecki, IEEE Computational Intelligence Magazine, 2015
    """
    def __init__(self, h, C=10000, random_state=666):
        super(TWELM, self).__init__(h=h, C=C, f=tanimoto, random_state=random_state, balanced=True)

