import numpy as np


def cov_to_inf(X, domain):
    """
    Converts covariance form to influence diagram form

    Parameters:
    X (numpy.ndarray): Gaussian distribution (covariance matrix)
    domain (int): the number of rows and columns of X (assuming X is square)

    Returns:
    B (numpy.ndarray): An n x n matrix of Gaussian influence diagram arc coefficients which is strictly upper triangular
    V (numpy.ndarray): An n x 1 matrix of Gaussian influence diagram conditional variances with non-negative entries (including inf)
    P (numpy.ndarray): The (Moore-Penrose generalized) inverse of X
    """

    P = np.zeros((domain, domain))
    V = np.zeros(domain)
    B = np.zeros((domain, domain))

    # Initialize first row
    B[0, 0] = 0
    V[0] = X[0, 0]

    if V[0] == 0 or V[0] == np.inf:
        P[0, 0] = 0
    else:
        P[0, 0] = 1 / V[0]

    # Iterate through the domain
    for j in range(1, domain):
        B[j, j] = 0
        B[j, :j] = 0
        B[:j, j] = P[:j, :j].dot(X[:j, j])

        if X[j, j] == np.inf:
            V[j] = np.inf
        else:
            V[j] = X[j, j] - np.dot(X[j, :j], B[:j, j])
            V[j] = max(V[j], 0)

        if V[j] == 0 or V[j] == np.inf:
            P[j, j] = 0
            P[j, :j] = 0
            P[:j, j] = 0
        else:
            P[j, j] = 1 / V[j]
            for k in range(j):
                temp = P[j, j] * B[k, j]
                if k != 0:
                    P[k, :k] += temp * B[:k, j]
                    P[:k, k] = P[k, :k]
                P[k, k] += temp * B[k, j]
            P[j, :j] = -P[j, j] * B[:j, j]
            P[:j, j] = P[j, :j]

    return B, V.reshape(-1, 1), P


def get_Kalman(B, n, j):
    K = np.zeros(n)
    # First initialize the first element
    K[0] = B[n + j, 0]

    # Then, recursively calculate each row of K
    for n_idx in range(1, n):
        # Direct term
        direct_term = B[n + j, n_idx]

        # Summation term
        summation_term = 0
        for i in range(n_idx):  # Summation from 1 to n-1 (n_idx, r_idx are zero-based)
            summation_term += K[i] * B[i, n_idx]

        # Compute K[n, r]
        K[n_idx] = direct_term + summation_term

    return K