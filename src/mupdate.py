import numpy as np

from src.cov_to_inf import cov_to_inf
from src.evidence import evidence


def mupdate(k, Z, u, B_or_sigma, V, R, H, h=None):
    """
    Mupdate
    Measurement update for measurement Z(k)

    Inputs:
    k - desired discrete time
    Z - p x 1 vector of measurement values at discrete time k
    u - n x 1 vector that represents the mean of the state X(k) at discrete time k
    B_or_sigma - If k == 0, n x n covariance matrix of state X(k) at discrete time k.
                 If k != 0, n x n matrix of Gaussian influence diagram arc coefficients of state X(k).
    V - If k != 0, n x 1 vector of Gaussian influence diagram conditional variances.
        If k == 0, this input is ignored.
    R - p x p diagonal measurement noise covariance matrix R
    H - p x n measurement matrix at discrete time k

    Outputs:
    u - updated n x 1 vector representing the mean of the state X(k+1)
    V - updated n x 1 vector of Gaussian influence diagram conditional variances of state X(k+1)
    B - updated n x n matrix of Gaussian influence diagram arc coefficients of state X(k+1)
    """

    # Determine dimensions of V and Z
    domain = V.shape[0]
    n = domain
    p = Z.shape[0]

    if k == 0:
        B, V, P = cov_to_inf(B_or_sigma, domain)

    # Prepare intermediate values for update
    X1 = np.array(Z).reshape(-1, 1)
    n0 = n
    n1 = p
    n2 = 0
    u_new = np.vstack((u, H @ u))
    if h is not None:
        u_new = np.vstack((u, h))

    # Construct V_new to match MATLAB structure
    V_new = np.vstack((V, np.diag(R).reshape(-1, 1)))

    # Build the B_new matrix
    Opn = np.zeros((p, n))
    Opp = np.zeros((p, p))
    B_new = np.block([[B, H.T], [Opn, Opp]])

    V_new = V_new.T
    V_new = V_new.flatten()
    u_new = u_new.T
    u_new = u_new.flatten()
    X1 = X1.T
    X1 = X1.flatten()
    du = u_new.copy()

    # Update using Evidence function
    u, B, V, K = evidence(u_new, B_new, V_new, X1, n0, n1, n2, du)

    # Return only the relevant portions
    u = u[:n]
    B = B[:n, :n]
    V = V[:n]

    return u, V, B, K