import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Attraction factor of CorInfoMax Loss: MSE loss calculation from outputs of the projection network, z1 (NXD) from 
    the first branch and z2 (NXD) from the second branch. Returns loss part comes from attraction factor (mean squared error).
    """
    return F.mse_loss(z1, z2)



class CovarianceLoss(nn.Module):
    """Big-bang factor of CorInfoMax Loss: loss calculation from outputs of the projection network,
    z1 (NXD) from the first branch and z2 (NXD) from the second branch. Returns loss part comes from bing-bang factor.
    """
    def __init__(self, args):
        super(CovarianceLoss, self).__init__()
        sizes = [512] + list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.R2 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) 
        self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) 
        self.new_R2 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) 
        self.new_mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) 
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        la_R = self.la_R
        la_mu = self.la_mu

        N, D = z1.size()

        # mean estimation
        mu_update1 = torch.mean(z1, 0)
        mu_update2 = torch.mean(z2, 0)
        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu_update1)
        self.new_mu2 = la_mu*(self.mu2) + (1-la_mu)*(mu_update2)

        # covariance matrix estimation
        z1_hat =  z1 - self.new_mu1
        z2_hat =  z2 - self.new_mu2
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N
        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

        # loss calculation 
        cov_loss = - (torch.logdet(self.new_R1 + self.R_eps) + torch.logdet(self.new_R2 + self.R_eps)) / D

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.mu1 = self.new_mu1.detach()
        self.R2 = self.new_R2.detach()
        self.mu2 = self.new_mu2.detach()

        return cov_loss

    def save_eigs(self) -> np.array: 
        with torch.no_grad():
            R_eig = torch.linalg.eigvals(self.R1).unsqueeze(0)
            self.R_eigs = torch.cat((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
        return R_eig_arr 

