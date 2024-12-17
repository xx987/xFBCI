import numpy as np

# from ApproximateInference import SG_MCMC_Infer,likelihood_p_k
from ApproxInfer import SG_MCMC_Infer  # ,likelihood_p_k

from jax.scipy.stats import multivariate_normal
from numpy import linalg as LA
import pyreadr
import optax
import jax
import jax.numpy as jnp


class EPClient:
    def __init__(self, eta_global,
                 Lambda_global,
                 eta_local,
                 Lambda_local,
                 delta=0.2):

        self.eta_global = eta_global
        self.Lambda_global = Lambda_global
        self.eta_local = eta_local
        self.Lambda_local = Lambda_local
        self.delta = delta


    def cavity_distribution(self):

        eta_cavity = self.eta_global - self.eta_local
        Lambda_cavity = self.Lambda_global - self.Lambda_local

        return eta_cavity, Lambda_cavity



    def update_local_distribution(self, eta_est_up, Lambda_est_up):


        delta_eta_k =eta_est_up - self.eta_global
        delta_Lambda_k = Lambda_est_up - self.Lambda_global


        eigenvalues, eigenvectors = np.linalg.eigh(delta_Lambda_k)
        epsilon = np.random.uniform(0.001, 0.01)  # 1e-3
        # Replace negative eigenvalues with a small positive number (epsilon)
        eigenvalues = np.where(eigenvalues > 0, eigenvalues, epsilon)

        # Reconstruct the matrix
        r_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


        return delta_eta_k, r_matrix

    def get_localnew_parameters(self, eta_est_gn, Lambda_est_gn):

        dk, dlk = self.update_local_distribution(eta_est_gn, Lambda_est_gn)
        e = self.eta_local
        l = self.Lambda_local
        new_elocal = e + self.delta * dk
        new_llocal = l + self.delta * dlk
        return dk, dlk, new_elocal, new_llocal



