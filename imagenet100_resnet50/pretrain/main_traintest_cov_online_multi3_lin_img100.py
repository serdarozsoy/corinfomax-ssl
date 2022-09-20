
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loss import CovarianceLossv2 as CovarianceLoss
from loss import invariance_loss

from model_base_resnet import CovModel6 as CovModel, LinModel
from metrics import correct_top_k, grad_norm

from pretrain_base_dist_multigpulinear import pretrain_cov_dist, Covdet
from linear_base2 import linear_train, linear_test
from lin_classifier import LinClassifier
import parsing_file_dist as parsing_file
import data_utils_dist as data_utils
import optim_utils3 as optim_utils
import save_utils_con_multi as save_utils
import save_utils_linear 

import copy
import numpy as np
import random
import os

from distributed import init_distributed_mode

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist



def train_test(args):
    
    set_seed(10)
    #torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    gpu = torch.device(args.device)

    print(f"============== ARGS ==============\n\n{args}\n\n==================================")

    pretrain_data, lin_train_data, lin_test_data  = data_utils.make_data(args.dataset, args.subset, args.subset_type)
    
    sampler = torch.utils.data.distributed.DistributedSampler(pretrain_data, shuffle=True)
    print(args.batch_size % args.world_size, args.batch_size , args.world_size)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size


    pre_train_loader = DataLoader(pretrain_data, batch_size=per_device_batch_size, num_workers=args.n_workers, sampler=sampler, pin_memory=True, drop_last=True)  


    lin_sampler = torch.utils.data.distributed.DistributedSampler(lin_train_data, shuffle=True)
    #sampler2 = torch.utils.data.distributed.DistributedSampler(lin_test_data, shuffle=False)
    assert args.lin_batch_size % args.world_size == 0
    per_device_lin_batch_size = args.lin_batch_size // args.world_size
    lin_train_loader = DataLoader(lin_train_data, batch_size=per_device_lin_batch_size, num_workers=args.n_workers, sampler=lin_sampler, pin_memory=True)
    #lin_test_loader = DataLoader(lin_test_data, batch_size=per_device_lin_batch_size, num_workers=args.n_workers, sampler=lin_sampler, pin_memory=False)
    
    lin_test_sampler = torch.utils.data.distributed.DistributedSampler(lin_test_data, shuffle=False)
    lin_test_loader = DataLoader(lin_test_data, batch_size=args.lin_batch_size, num_workers=args.n_workers, shuffle=False, sampler=lin_test_sampler, pin_memory=True)

    # Initialize loss and classifier 
    covariance_loss = CovarianceLoss(args)


    # Model and optimizer setup 
    pre_model = CovModel(args).cuda(gpu)
    lin_classifier = LinClassifier(args, offline=True).cuda(gpu)
    # Model and loss
    pre_model = Covdet(pre_model, covariance_loss, args, lin_classifier)
    pre_model = nn.SyncBatchNorm.convert_sync_batchnorm(pre_model)
    pre_model = DDP(pre_model, device_ids=[gpu], find_unused_parameters=True)



    pre_optimizer = optim_utils.make_optimizer(pre_model, args, pretrain=True)
    pre_scheduler = optim_utils.make_scheduler(pre_optimizer, args, pretrain=True)

    lin_optimizer = optim_utils.make_optimizer(lin_classifier, args, pretrain=False)
    lin_scheduler = optim_utils.make_scheduler(lin_optimizer, args, pretrain=False)


    if args.rank == 0:
        save_results = save_utils.SaveResults(args)
        writer, save_name_linear = save_results.create_results()
        save_results.save_parameters()



    for epoch in range(1, args.epochs+1):
        sampler.set_epoch(epoch)
        lin_sampler.set_epoch(epoch)
        lin_test_sampler.set_epoch(epoch)

        # Training 
        train_loss, train_cov_loss, train_sim_loss, break_code = pretrain_cov_dist(pre_model, pre_train_loader, pre_optimizer, invariance_loss, epoch)
        if break_code:
            test_acc1 = 0.0
            break
        # get learning rate from optimizer
        pre_curr_lr = pre_optimizer.param_groups[0]['lr']
        total_norm = grad_norm(pre_model)

        if args.rank == 0:
            save_results.update_results(train_loss, train_cov_loss, train_sim_loss, pre_curr_lr, total_norm)
            R_eigs = save_results.save_eigs(covariance_loss, epoch)
            save_utils.update_tensorboard(writer, train_loss, train_cov_loss, train_sim_loss, pre_curr_lr, total_norm, R_eigs, epoch)
            save_results.save_stats(epoch)

            writer.flush()

            if (epoch % 100 == 0)|(epoch == args.epochs):
                save_results.save_model(pre_model, pre_optimizer, train_loss, epoch)

        # Scheduler action
        if args.pre_scheduler == "None":
            pass
        elif args.pre_scheduler == "reduce_plateau":
            pre_scheduler.step(train_loss)
        else:
            pre_scheduler.step()

        linear_period = args.epochs


        # Training 
        linear_loss, linear_acc1, linear_acc5 = linear_train(pre_model, lin_train_loader, lin_optimizer, epoch)


        # get learning rate from optimizer
        lin_curr_lr = lin_optimizer.param_groups[0]['lr']
        total_lin_norm = grad_norm(lin_classifier)
        with torch.no_grad():
            # Testing
            test_loss, test_acc1, test_acc5 = linear_test(pre_model, lin_test_loader, epoch)

        # Model and optimizer setup 
        if args.rank == 0:
            save_epoch = epoch
            save_results.update_lin_results(linear_loss, linear_acc1, linear_acc5, lin_curr_lr, test_loss, test_acc1, test_acc5, total_lin_norm)
            save_utils.update_lin_tensorboard(writer, linear_loss, linear_acc1, linear_acc5, lin_curr_lr, test_loss, test_acc1, test_acc5, total_lin_norm, save_epoch)
            save_results.save_lin_stats(save_epoch,)

            writer.flush()

        # Scheduler action
        if args.lin_scheduler == "None":
            pass
        elif args.lin_scheduler == "reduce_plateau":
            lin_scheduler.step(linear_loss)
        else:
            lin_scheduler.step()


    if args.rank == 0:
        save_results.save_model_resnet(pre_model)    
        

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = parsing_file.create_parser()
    arguments = parser.parse_args()

    arguments.lin_epochs = arguments.epochs
    train_test(arguments)

