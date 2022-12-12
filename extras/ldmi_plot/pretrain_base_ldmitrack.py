import torch
from tqdm import tqdm
from loss import invariance_loss
import numpy as np

def pretrain_cov(net, data_loader, covariance_loss,  train_optimizer, cov_loss_weight, sim_loss_weight, epoch):
    net.train()
    total_loss, loss_sim, loss_cov, total_num, train_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    break_code = False
    for data_tuple in train_bar:
        (pos_1, pos_2), target = data_tuple
        pos_1, pos_2 = pos_1.cuda(), pos_2.cuda()
        # Forward prop of the model for both inputs
        z1, z2  = net(pos_1, pos_2)
        # Batchsize before concat
        batchsize_bc = z1.shape[0]
        # Call similarity and covariance losses
        cov_loss = covariance_loss(z1, z2)
        sim_loss = invariance_loss(z1, z2) 
        loss = (sim_loss_weight * sim_loss) + (cov_loss_weight * cov_loss)

        # Backpropagation part
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # Accumulating number of examples, losses and correct predictions
        total_num += batchsize_bc

        total_loss += loss.item() * batchsize_bc
        loss_cov += cov_loss.item() * batchsize_bc
        loss_sim += sim_loss.item() * batchsize_bc


        if np.isnan(total_loss):
            break_code = True
            print("Code is breaked due to NaN")
            break


        # This bar is used for live tracking on command line (batch_size -> batchsize_bc: to show current batchsize )
        train_bar.set_description('Train Epoch: [{}] Loss: {:.4f} Cov_loss: {:.4f} Sim_loss: {:.4f}'.format(\
                epoch, total_loss / total_num, loss_cov / total_num, loss_sim / total_num))
    
    return total_loss/total_num, loss_cov/total_num,  loss_sim/total_num, break_code


def pretrain_cov_ldmitrack(net, data_loader, covariance_loss, ldmi_tracking, train_optimizer, cov_loss_weight, sim_loss_weight, epoch):
    net.train()
    total_ldmi, total_loss, loss_sim, loss_cov, total_num, train_bar = 0.0, 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    break_code = False
    for data_tuple in train_bar:
        (pos_1, pos_2), target = data_tuple
        pos_1, pos_2 = pos_1.cuda(), pos_2.cuda()
        # Forward prop of the model for both inputs
        z1, z2  = net(pos_1, pos_2)
        # Batchsize before concat
        batchsize_bc = z1.shape[0]
        # Call similarity and covariance losses
        cov_loss = covariance_loss(z1, z2)
        sim_loss = invariance_loss(z1, z2)
        with torch.no_grad():
            batch_ldmi = ldmi_tracking(z1,z2)
        loss = (sim_loss_weight * sim_loss) + (cov_loss_weight * cov_loss)

        # Backpropagation part
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # Accumulating number of examples, losses and correct predictions
        total_num += batchsize_bc

        total_ldmi += batch_ldmi.item() * batchsize_bc
        total_loss += loss.item() * batchsize_bc
        loss_cov += cov_loss.item() * batchsize_bc
        loss_sim += sim_loss.item() * batchsize_bc


        if np.isnan(total_loss):
            break_code = True
            print("Code is breaked due to NaN")
            break


        # This bar is used for live tracking on command line (batch_size -> batchsize_bc: to show current batchsize )
        train_bar.set_description('Train Epoch: [{}] LDMI: {:.4f} Loss: {:.4f} Cov_loss: {:.4f} Sim_loss: {:.4f}'.format(\
                epoch, total_ldmi/total_num, total_loss / total_num, loss_cov / total_num, loss_sim / total_num))
    
    return  total_ldmi/total_num, total_loss/total_num, loss_cov/total_num,  loss_sim/total_num, break_code