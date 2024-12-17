import numpy as np
import torch
import torch.optim as optim
#from FederatedLearning import eta_local
#from GaussianEPClient import EPClient

class EPServer:
    def __init__(self, e_global, l_global, sum_eta, sum_cov):


        self.e_global = e_global
        self.l_global = l_global
        self.sum_eta = sum_eta
        self.sum_cov = sum_cov
    def InfServer(self):
        """Input: e_global, l_global: current global eta and lambda
        sum_mean, sum_cov: list,
        the list contains all of delta-parameter of all clients.

        Output: Updated global eta and Lambda

        """

        sum_e = np.sum(self.sum_eta,axis=0)
        sum_l = np.sum(self.sum_cov,axis=0)
        new_e = np.array(self.e_global)
        new_l =  np.array(self.l_global)

        new_eg = new_e + 0.2 * sum_e  # 应用阻尼系数 delta 来更新 eta_k
        new_lg = new_l + 0.2 * sum_l

        return  new_eg, new_lg







