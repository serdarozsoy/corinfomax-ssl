import torch
from tqdm import tqdm
from loss import invariance_loss
import numpy as np

import torch.distributed as dist
from metrics import correct_top_k
import torch.nn as nn
import torch.nn.functional as F

from distributed import get_world_size

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = []
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

class Covdet(nn.Module):
    def __init__(self, net, loss, args, linear=None):
        super(Covdet, self).__init__()
        self.net = net
        self.loss = loss
        self.sim_loss_weight = args.sim_loss_weight
        self.cov_loss_weight = args.cov_loss_weight
        self.linear=linear

    # Does both regular forward pass as well as linear layer forward pass
    def forward(self, pos_1, pos_2=None, invariance_loss=None, targets=None):
        linear_pass = pos_2==None and invariance_loss==None

        if linear_pass:
            return self.linear_forward(pos_1, targets)
        else:
            return self.covdet_forward(pos_1, pos_2, invariance_loss)

    def linear_forward(self, pos_1, targets):
        training_flag = self.training

        # set backbone to eval mode
        self.net.eval()

        # Get backbone outputs logits
        with torch.no_grad():
            backbone_logits = self.net.backbone(pos_1)

        # Revert net training flag to original
        self.net.train(training_flag)

        # Calculate derivatives only for linear head
        logits = self.linear(backbone_logits)

        # Gather all logits
        logits = torch.cat(FullGatherLayer.apply(logits), dim=0)
        targets = torch.cat(FullGatherLayer.apply(targets), dim=0)

        # Loss 
        linear_loss_1 = F.cross_entropy(logits, targets)

        # Number of correct predictions
        linear_correct_1, linear_correct_5 = correct_top_k(logits, targets, top_k=(1, 5))

        return linear_loss_1, linear_correct_1, linear_correct_5

    def covdet_forward(self, pos_1, pos_2, invariance_loss):
        z1, z2  = self.net(pos_1, pos_2)
        batchsize_bc = z1.shape[0]

        z1 = torch.cat(FullGatherLayer.apply(z1), dim=0)
        z2 = torch.cat(FullGatherLayer.apply(z2), dim=0)

        sim_loss = invariance_loss(z1, z2)        
        """
        # Gather all z1s, z2s
        z1 = torch.cat(FullGatherLayer.apply(z1), dim=0)
        z2 = torch.cat(FullGatherLayer.apply(z2), dim=0)
        """
        # Prevent autocast from potentially using lower precision within covdet loss
        # with torch.cuda.amp.autocast(enabled=False):
        cov_loss = self.loss(z1, z2)
        
        loss = self.sim_loss_weight * sim_loss + self.cov_loss_weight * cov_loss

        return loss, sim_loss.detach(), cov_loss.detach()




def pretrain_cov_dist(net, data_loader, train_optimizer, invariance_loss, epoch, scaler, grad_accumulation_steps=0, amp=False, half=False):
    net.train()
    total_loss, loss_sim, loss_cov, total_num, train_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    break_code = False
    steps = 0
    train_optimizer.zero_grad()

    for data_tuple in train_bar:
        steps += 1
        (pos_1, pos_2), target = data_tuple
        #print(pos_1.size())
        pos_1, pos_2 = pos_1.cuda(), pos_2.cuda()
        # print("pos_1.sum(), pos_1.mean()", pos_1.sum(), pos_1.mean())
        # print("pos_2.sum(), pos_2.mean()", pos_2.sum(), pos_2.mean())
        if half:
            pos_1, pos_2 = pos_1.half(), pos_2.half()
        # Forward prop of the model for both inputs
        batchsize_bc = pos_1.shape[0] * get_world_size()
        # Sync gradients only if update being performed
        if steps % grad_accumulation_steps == 0:
            with torch.cuda.amp.autocast(enabled=amp):
                loss, sim_loss, cov_loss = net(pos_1, pos_2, invariance_loss)    
            loss_bef_scale = loss.item()
            # print("loss, sim_loss, cov_loss", loss, sim_loss, cov_loss)
            # exit(0)
            # Backpropagation part
            scaler.scale(loss).backward()
            scaler.step(train_optimizer)
            scaler.update()
            train_optimizer.zero_grad()
        else: 
            # Otherwise, prevent gradient syncing to speed up the iteration
            with net.no_sync():
                with torch.cuda.amp.autocast(enabled=amp):
                    loss, sim_loss, cov_loss = net(pos_1, pos_2, invariance_loss)    
                loss_bef_scale = loss.item()
                # Backpropagation part
                scaler.scale(loss).backward()
            

        # Accumulating number of examples, losses and correct predictions
        total_num += batchsize_bc
        total_loss += loss_bef_scale * batchsize_bc 
        loss_cov += cov_loss.item() * batchsize_bc 
        loss_sim += sim_loss.item() * batchsize_bc 
 
        if np.isnan(total_loss):
            break_code = True
            print("Code is breaked due to NaN")
            break


        # This bar is used for live tracking on command line (batch_size -> batchsize_bc: to show current batchsize )
        train_bar.set_description('Train Epoch: [{}] Loss: {:.4f} Cov_loss: {:.4f} Sim_loss: {:.4f}'.format(\
                epoch, total_loss / total_num, loss_cov / total_num, loss_sim / total_num))

    # This might be broken by the no_sync optimization
    # # Any leftover steps
    # if steps % grad_accumulation_steps != 0:
    #     scaler.step(train_optimizer)
    #     scaler.update()
    #     train_optimizer.zero_grad()

    return total_loss/total_num, loss_cov/total_num,  loss_sim/total_num, break_code


def pretrain_exp(net, data_loader, covariance_loss,  train_optimizer, cov_loss_weight, sim_loss_weight, epoch):
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
        cov_loss = torch.tensor(0.0) #covariance_loss(z1, z2)
        # dist = z1-z2
        sim_loss = invariance_loss(z1, z2) #(torch.norm(dist)**2) / (dist.shape[0]*dist.shape[1])# invariance_loss_sum(z1, z2) 
        loss = (sim_loss_weight * sim_loss) # + (cov_loss_weight * cov_loss)

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




def pretrain_ldmi(net, data_loader, ldmi_loss,  train_optimizer, cov_loss_weight, epoch):
    net.train()
    total_loss, loss_cov, loss_cov11, loss_cov12, loss_cov2, total_num, train_bar = 0.0, 0.0, 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    break_code = False
    for data_tuple in train_bar:
        (pos_1, pos_2), target = data_tuple
        pos_1, pos_2 = pos_1.cuda(), pos_2.cuda()
        # Forward prop of the model for both inputs
        z1, z2  = net(pos_1, pos_2)
        # Batchsize before concat
        batchsize_bc = z1.shape[0]
        #print(z1.size())
        # Call similarity and covariance losses
        cov_loss, cov11_loss, cov12_loss, cov2_loss = ldmi_loss(z1, z2)
        loss = cov_loss_weight * cov_loss

        # Backpropagation part
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # Accumulating number of examples, losses and correct predictions
        total_num += batchsize_bc
        total_loss += loss.item() * batchsize_bc

        loss_cov += cov_loss.item() * batchsize_bc
        loss_cov11 += cov11_loss.item() * batchsize_bc
        loss_cov12 += cov12_loss.item() * batchsize_bc
        loss_cov2 += cov2_loss.item() * batchsize_bc

        if np.isnan(total_loss):
            break_code = True
            print("Code is breaked due to NaN")
            break


        # This bar is used for live tracking on command line (batch_size -> batchsize_bc: to show current batchsize )
        train_bar.set_description('Train Epoch: [{}] Loss: {:.4f} Cov_loss: {:.4f} Cov_loss11: {:.4f} Cov_loss12: {:.4f} Cov_loss2: {:.4f}'.format(\
                epoch, total_loss / total_num, loss_cov / total_num, loss_cov11 / total_num, loss_cov12 / total_num, loss_cov2 / total_num))
    
    return total_loss/total_num, loss_cov/total_num,  loss_cov11/total_num, loss_cov12/total_num, loss_cov2/total_num, break_code

