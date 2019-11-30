import numpy as np
from numpy.linalg import slogdet, det, solve
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_digits
from scipy.stats import multivariate_normal


def E_step(X, pi, mu, sigma):

    N = X.shape[0] # number of objects
    C = pi.shape[0] # number of clusters
    d = mu.shape[1] # dimension of each object
    gamma = np.zeros((N, C)) # distribution q(T)

    for _c in range(C):
        gamma[:,_c] = multivariate_normal.pdf(X,mean=mu0[_c],cov=sigma0[_c])*pi[_c]
    Z = np.sum(gamma,1)
    Z = np.reshape(Z,(-1,1))
    gamma = gamma/Z
    return gamma


def M_step(X, gamma):

    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object
    mu  = np.zeros((C, d))
    sigma = np.zeros((C,d,d))
    pi = np.zeros(C)
    ### YOUR CODE HERE
    for _c in range(C):
      x = X * np.reshape(gamma[:,_c],(-1,1))
      mu[_c] = np.sum(x,0)/np.sum(gamma[:,_c],0)
      X1 = (X - mu[_c])
      X11 = np.dot(np.transpose(X1),(X1*np.reshape(gamma[:,_c],(-1,1))))
      sigma[_c] = X11/np.sum(gamma[:,_c],0)
      pi[_c] = np.sum(gamma[:,_c],0)/N

    return pi, mu, sigma



def compute_vlb(X, pi, mu, sigma, gamma):   # Variational LowerBound

    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object
    ll = np.zeros((N,C))
    for _c in range(C):
        ll[:,_c] = multivariate_normal.logpdf(X,mean=mu[_c],cov=sigma[_c])


    ll = (ll + np.log(pi))*gamma - gamma*np.log(np.maximum(gamma,1e-8))
    return np.sum(ll)




def train_EM(X, C, rtol=1e-3, max_iter=100, restarts=10):

    N = X.shape[0] # number of objects
    d = X.shape[1] # dimension of each object
    best_loss = -1e10
    best_pi = pi0
    best_mu = mu0
    best_sigma = sigma0

    for _ in range(restarts):
        losses = []
        try:
            for _ in range(max_iter):
              gamma = E_step(X, best_pi, best_mu, best_sigma)
              pi, mu, sigma = M_step(X, gamma)
              loss = compute_vlb(X, pi, mu, sigma, gamma)
              losses.append(loss)
              if loss >= best_loss:
                  best_pi = pi
                  best_mu = mu
                  best_sigma = sigma
                  best_loss = loss

        except np.linalg.LinAlgError:
              print("Singular matrix: components collapsed")
              pass


    return best_loss, best_pi, best_mu, best_sigma
