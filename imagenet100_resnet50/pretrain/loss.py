import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CovarianceLoss(nn.Module):
    def __init__(self, args):
        super(CovarianceLoss, self).__init__()
        sizes = [512] + list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_mu = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        la_R = self.la_R 
        la_mu = self.la_mu

        N, D = z1.size()
        z = torch.cat((z1, z2), 0)

        #z_hat =  z - self.mu   #Previous version
        #R_update = (z_hat.T @ z_hat) / (2*N)   #Previous version
        mu_update = torch.mean(z, 0)

        
        #self.new_R = la_R*(self.R) + (1-la_R)*(R_update)   #Previous version
        self.new_mu = la_mu*(self.mu) + (1-la_mu)*(mu_update)

        z_hat =  z - self.new_mu
        R_update = (z_hat.T @ z_hat) / (2*N)
        self.new_R = la_R*(self.R) + (1-la_R)*(R_update)

        cov_loss = -torch.logdet(self.new_R + self.R_eps) / D

        # This is required because new_R updated with backward.
        self.R = self.new_R.detach()
        self.mu = self.new_mu.detach()

        return cov_loss

    def save_eigs(self) -> np.array: 
        with torch.no_grad():
            R_eig = torch.linalg.eigvals(self.R).unsqueeze(0)
            self.R_eigs = torch.cat((self.R_eigs, R_eig), 0)
            #self.R_eigs = torch.stack((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
            # R_eig_arr = np.sort(R_eig_arr =)[::-1] # sorted eigenvalues
        return R_eig_arr 


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """
    return F.mse_loss(z1, z2)


class LDMILoss(nn.Module):
    def __init__(self, args):
        super(LDMILoss, self).__init__()
        sizes = [512] + list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.R2 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.Re = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.R12 = torch.zeros((proj_output_dim,proj_output_dim) , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim,proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=True)
        self.new_R2 = torch.zeros((proj_output_dim,proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=True)
        self.new_Re12 = torch.zeros((proj_output_dim,proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=True)
        self.new_Re21 = torch.zeros((proj_output_dim,proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=True)
        self.new_R12 = torch.zeros((proj_output_dim,proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=True)
        self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.Re_eigs = torch.linalg.eigvals(self.Re).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps_weight2 = args.R_eps_weight2
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.R_eps2 = self.R_eps_weight2*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        la_R = self.la_R 
        la_mu = self.la_mu

        N, D = z1.size()
        z = torch.cat((z1, z2), 0)
        E = z1-z2
        

        z1_hat =  z1 - self.mu1
        z2_hat =  z2 - self.mu2
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N
        R12_update = (z1_hat.T @ z2_hat) / N
        Re_update = (E.T @ E) / N
        mu1_update = torch.mean(z1, 0)
        mu2_update = torch.mean(z2, 0)

        
        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)
        self.new_R12 = la_R*(self.R12) + (1-la_R)*(R12_update)
        self.new_Re12= self.new_R1-self.new_R12@torch.linalg.inv(self.new_R2+self.R_eps)@self.new_R12.T
        self.new_Re21= self.new_R2-self.new_R12.T@torch.linalg.inv(self.new_R1+self.R_eps)@self.new_R12
        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu1_update)
        self.new_mu2 = la_mu*(self.mu2) + (1-la_mu)*(mu2_update)
        
        cov_loss11=torch.logdet(self.new_R1 + self.R_eps)/D
        cov_loss12=torch.logdet(self.new_R2 + self.R_eps)/D
        cov_loss2=torch.logdet(self.new_Re12 + self.R_eps2)/D+torch.logdet(self.new_Re21 + self.R_eps2)/D

        cov_loss = cov_loss2-cov_loss11-cov_loss12

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.mu1 = self.new_mu1.detach()
        self.R2 = self.new_R2.detach()
        self.mu2 = self.new_mu2.detach()
        self.R12 = self.new_R12.detach()

        return cov_loss, cov_loss11, cov_loss12,cov_loss2
        

    def save_eigs(self) -> np.array: 
        R_eig = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eigs = torch.cat((self.R_eigs, R_eig), 0)
        #self.R_eigs = torch.stack((self.R_eigs, R_eig), 0)
        R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
        # R_eig_arr = np.sort(R_eig_arr =)[::-1] # sorted eigenvalues
        return R_eig_arr 


class CovarianceLossv2(nn.Module):
    def __init__(self, args):
        super(CovarianceLossv2, self).__init__()
        sizes = [512] + list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.R2 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_R2 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        la_R = self.la_R
        la_mu = self.la_mu


        N, D = z1.size()
        #z = torch.cat((z1, z2), 0)

        #z_hat =  z - self.mu   #Previous version
        #R_update = (z_hat.T @ z_hat) / (2*N)   #Previous version
        mu_update1 = torch.mean(z1, 0)
        mu_update2 = torch.mean(z2, 0)

        
        #self.new_R = la_R*(self.R) + (1-la_R)*(R_update)   #Previous version
        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu_update1)
        self.new_mu2 = la_mu*(self.mu2) + (1-la_mu)*(mu_update2)

        z1_hat =  z1 - self.new_mu1
        z2_hat =  z2 - self.new_mu2
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N
        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

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
            #self.R_eigs = torch.stack((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
            # R_eig_arr = np.sort(R_eig_arr =)[::-1] # sorted eigenvalues
        return R_eig_arr 



class CovarianceLossv4(nn.Module):
    def __init__(self, args):
        super(CovarianceLossv4, self).__init__()
        sizes = [512] + list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        #self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.R2 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        #self.mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=True) # Changed to requires_grad=False
        #self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_R2 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=True) # Changed to requires_grad=False
        #self.new_mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        la_R = self.la_R

        N, D = z1.size()

        mu1_update = torch.mean(z1, 0)
        mu2_update = torch.mean(z2, 0)

        z1_hat =  z1 - mu1_update 
        z2_hat =  z2 - mu2_update 
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N

        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

        cov_loss = - (torch.logdet(self.new_R1 + self.R_eps) + torch.logdet(self.new_R2 + self.R_eps)) / D

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.R2 = self.new_R2.detach()

        return cov_loss

    def save_eigs(self) -> np.array: 
        with torch.no_grad():
            R_eig = torch.linalg.eigvals(self.R1).unsqueeze(0)
            self.R_eigs = torch.cat((self.R_eigs, R_eig), 0)
            #self.R_eigs = torch.stack((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
            #R_eig_arr = np.sort(R_eig_arr[i,:])[::-1] # sorted eigenvalues
        return R_eig_arr



class CovarianceLossv6(nn.Module):
    def __init__(self, args):
        super(CovarianceLossv6, self).__init__()
        sizes = [512] + list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        #self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.R2 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        #self.mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        #self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_R2 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        #self.new_mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        la_R = self.la_R

        N, D = z1.size()

        mu1_update = torch.mean(z1, 0)
        mu2_update = torch.mean(z2, 0)

        z1_hat =  z1 - mu1_update 
        z2_hat =  z2 - mu2_update 
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N

        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

        cov_loss = - (torch.logdet(self.new_R1 + self.R_eps) + torch.logdet(self.new_R2 + self.R_eps)) / D 

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.R2 = self.new_R2.detach()

        return cov_loss

    def save_eigs(self) -> np.array: 
        with torch.no_grad():
            R_eig = torch.linalg.eigvals(self.R1).unsqueeze(0)
            self.R_eigs = torch.cat((self.R_eigs, R_eig), 0)
            #self.R_eigs = torch.stack((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
            #R_eig_arr = np.sort(R_eig_arr[i,:])[::-1] # sorted eigenvalues
        return R_eig_arr

class CovarianceLossv8(nn.Module):
    def __init__(self, args):
        super(CovarianceLossv8, self).__init__()
        sizes = [512] #+ list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.R2 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_R2 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        la_R = self.la_R
        la_mu = self.la_mu


        N, D = z1.size()
        #z = torch.cat((z1, z2), 0)

        #z_hat =  z - self.mu   #Previous version
        #R_update = (z_hat.T @ z_hat) / (2*N)   #Previous version
        mu_update1 = torch.mean(z1, 0)
        mu_update2 = torch.mean(z2, 0)

        
        #self.new_R = la_R*(self.R) + (1-la_R)*(R_update)   #Previous version
        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu_update1)
        self.new_mu2 = la_mu*(self.mu2) + (1-la_mu)*(mu_update2)

        z1_hat =  z1 - self.new_mu1
        z2_hat =  z2 - self.new_mu2
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N
        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

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
            #self.R_eigs = torch.stack((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
            # R_eig_arr = np.sort(R_eig_arr =)[::-1] # sorted eigenvalues
        return R_eig_arr 



class CovarianceLossv10(nn.Module):
    def __init__(self, args):
        super(CovarianceLossv10, self).__init__()
        sizes = [512] + list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        la_R = self.la_R
        la_mu = self.la_mu


        N, D = z1.size()
        #z = torch.cat((z1, z2), 0)

        #z_hat =  z - self.mu   #Previous version
        #R_update = (z_hat.T @ z_hat) / (2*N)   #Previous version
        mu_update1 = torch.mean(z1, 0)

        #self.new_R = la_R*(self.R) + (1-la_R)*(R_update)   #Previous version
        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu_update1)

        z1_hat =  z1 - self.new_mu1

        R1_update = (z1_hat.T @ z1_hat) / N

        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)

        cov_loss = - torch.logdet(self.new_R1 + self.R_eps) / D

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.mu1 = self.new_mu1.detach()

        return cov_loss

    def save_eigs(self) -> np.array: 
        with torch.no_grad():
            R_eig = torch.linalg.eigvals(self.R1).unsqueeze(0)
            self.R_eigs = torch.cat((self.R_eigs, R_eig), 0)
            #self.R_eigs = torch.stack((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
            # R_eig_arr = np.sort(R_eig_arr =)[::-1] # sorted eigenvalues
        return R_eig_arr 


class CovarianceLossv12(nn.Module):
    def __init__(self, args):
        super(CovarianceLossv12, self).__init__()
        sizes = [512] + list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = args.R_ini*(torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False) + torch.randn((proj_output_dim,proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False))
        self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.R2 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_R2 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        la_R = self.la_R
        la_mu = self.la_mu


        N, D = z1.size()
        #z = torch.cat((z1, z2), 0)

        #z_hat =  z - self.mu   #Previous version
        #R_update = (z_hat.T @ z_hat) / (2*N)   #Previous version
        mu_update1 = torch.mean(z1, 0)
        mu_update2 = torch.mean(z2, 0)

        
        #self.new_R = la_R*(self.R) + (1-la_R)*(R_update)   #Previous version
        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu_update1)
        self.new_mu2 = la_mu*(self.mu2) + (1-la_mu)*(mu_update2)

        z1_hat =  z1 - self.new_mu1
        z2_hat =  z2 - self.new_mu2
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N
        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

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
            #self.R_eigs = torch.stack((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
            # R_eig_arr = np.sort(R_eig_arr =)[::-1] # sorted eigenvalues
        return R_eig_arr 


class CovarianceLossv20(nn.Module):
    def __init__(self, args):
        super(CovarianceLossv20, self).__init__()
        sizes = [512] + list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu1 = torch.zeros(proj_output_dim, device='cuda', requires_grad=False)
        self.R2 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu2 = torch.zeros(proj_output_dim,  device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_mu1 = torch.zeros(proj_output_dim, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_R2 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_mu2 = torch.zeros(proj_output_dim,  device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        #self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        la_R = self.la_R
        la_mu = self.la_mu


        N, D = z1.size()
        #z = torch.cat((z1, z2), 0)

        #z_hat =  z - self.mu   #Previous version
        #R_update = (z_hat.T @ z_hat) / (2*N)   #Previous version
        mu_update1 = torch.mean(z1, 0)
        mu_update2 = torch.mean(z2, 0)

        
        #self.new_R = la_R*(self.R) + (1-la_R)*(R_update)   #Previous version
        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu_update1)
        self.new_mu2 = la_mu*(self.mu2) + (1-la_mu)*(mu_update2)

        z1 =  z1 - self.new_mu1
        z2 =  z2 - self.new_mu2
        R1_update = (z1.T @ z1) / N
        R2_update = (z2.T @ z2) / N
        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

        cov_loss = - (torch.logdet(self.new_R1 + self.R_eps) + torch.logdet(self.new_R2 + self.R_eps)) / D

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.mu1 = self.new_mu1.detach()
        self.R2 = self.new_R2.detach()
        self.mu2 = self.new_mu2.detach()


        return cov_loss


class CovarianceLossv21(nn.Module):
    def __init__(self, args):
        super(CovarianceLossv21, self).__init__()
        sizes = [512] + list(map(int, args.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.R2 = args.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.new_R2 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) # Changed to requires_grad=False
        self.la_R = args.la_R
        self.la_mu = args.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        la_R = self.la_R
        la_mu = self.la_mu

        N, D = z1.size()
        #z = torch.cat((z1, z2), 0)

        R1_update = (z1.T @ z1) / N
        R2_update = (z2.T @ z2) / N
        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

        cov_loss = - (torch.logdet(self.new_R1 + self.R_eps) + torch.logdet(self.new_R2 + self.R_eps)) / D

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.R2 = self.new_R2.detach()


        return cov_loss

    def save_eigs(self) -> np.array: 
        with torch.no_grad():
            R_eig = torch.linalg.eigvals(self.R1).unsqueeze(0)
            self.R_eigs = torch.cat((self.R_eigs, R_eig), 0)
            #self.R_eigs = torch.stack((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
            # R_eig_arr = np.sort(R_eig_arr =)[::-1] # sorted eigenvalues
        return R_eig_arr 