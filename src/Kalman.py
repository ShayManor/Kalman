from src.inf_to_cov import inf_to_cov
from src.mupdate import mupdate
from src.tupdate import tupdate


class trackingKF:
    def __init__(self, F, H, state, state_covariance, process_noise):
        self.StateTransitionModel = F
        self.MeasurementModel = H
        self.State = state
        self.StateCovariance = state_covariance
        self.ProcessNoise = process_noise


def kalman(k, Z, u, X, V, R, H, Phi, gamma, Qk, Form, h=None):
    """
    Apply Kalman filter at time k.

    Parameters:
    k (int): Desired discrete time.
    Z (numpy.ndarray): Measurement values at discrete time k.
    u (numpy.ndarray): An n x 1 vector representing the mean of the state at time k.
    X (numpy.ndarray): If k=0, the covariance matrix of state at time k. If k≠0, the matrix of Gaussian influence diagram arc coefficients.
    V (numpy.ndarray): If k≠0, an n x 1 vector of Gaussian influence diagram conditional variances. Ignored if k=0.
    R (numpy.ndarray): The measurement noise covariance matrix R.
    H (numpy.ndarray): The measurement matrix at discrete time k.
    Phi (numpy.ndarray): The state transition matrix at time k.
    gamma (numpy.ndarray): The process noise matrix at time k.
    Qk (numpy.ndarray): The process noise covariance matrix.
    Form (int): Determines the output form (0 for ID form, 1 for covariance form).

    Returns:
    u (numpy.ndarray): The updated mean vector.
    B (numpy.ndarray): The updated state covariance matrix or influence diagram form matrix.
    V (numpy.ndarray): The updated vector of conditional variances.
    """

    # Get dimensions
    domain = X.shape[0]
    p = Z.shape[0]

    # Perform measurement update
    u, V, B, K = mupdate(k, Z, u, X, V, R, H, h)
    u_new = u[:domain]
    V_new = V[:domain]
    B_new = B[:domain, :domain]
    # K_new = K[:domain, :domain]
    # Perform time update
    u, B, V = tupdate(u_new, B_new, V_new, Phi, gamma, Qk)

    # Convert back to covariance form if required
    if Form == 1:
        B = inf_to_cov(V, B, domain)

    return u, B, V, K
