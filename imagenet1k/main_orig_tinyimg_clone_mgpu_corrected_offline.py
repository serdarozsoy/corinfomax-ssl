import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from loss import CovarianceLossv2 as CovarianceLoss, invariance_loss 
from model_base_resnet import CovModel6 as CovModel, LinModel
from metrics import correct_top_k, grad_norm
from distributed import init_distributed_mode

from pretrain_base_dist_multigpulinear import pretrain_cov_dist, Covdet
# Swap these
# from linear_base3 import linear_train, linear_test
from linear_base2 import linear_train, linear_test
# end swap these
from lin_classifier import LinClassifier
import parsing_file_dist as parsing_file
import data_utils_offline as data_utils 
import optim_utils
import save_utils_con_multi as save_utils

import copy
import numpy as np
import random
import os

from torch.nn.parallel import DistributedDataParallel as DDP

# To remove randomness if there is need
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_test(args):
    print(args)

    # If there is more than one CUDA device:
    # Change for DIST
    # torch.cuda.set_device(0)
    init_distributed_mode(args)
    gpu = torch.device(args.device)
    # gpu = torch.device(0)

    # args.world_size=1
    print(args)

    pretrain_data  = data_utils.make_data(args.dataset, args.subset, args.subset_type)
    # Swap these
    sampler = torch.utils.data.distributed.DistributedSampler(pretrain_data, shuffle=True)
    # sampler = torch.utils.data.distributed.DistributedSampler(pretrain_data, shuffle=True, num_replicas=1, rank=0)
    # Swap
    print("Per gpu batch size: ", args.batch_size // args.world_size, 
        "\nAll GPU Batchsize: ", args.batch_size , 
        "\nEffective batchsize (with gradaccum)", args.batch_size*args.grad_accumulation_steps, 
        "\nNum GPUs: ", args.world_size)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    pre_train_loader = DataLoader(pretrain_data, batch_size=per_device_batch_size, num_workers=args.n_workers, sampler=sampler, pin_memory=True, drop_last=True)

    # Initialize loss and classifier 
    covariance_loss = CovarianceLoss(args)

    # Model and optimizer setup 
    pre_model = CovModel(args).cuda(gpu)
    lin_classifier = LinClassifier(args, offline=True).cuda(gpu)
    pre_model = Covdet(pre_model, covariance_loss, args, lin_classifier)
    pre_model = nn.SyncBatchNorm.convert_sync_batchnorm(pre_model)

    # pre_optimizer = optim_utils.make_optimizer(pre_model, args, pretrain=True)
    pre_optimizer = optim_utils.make_optimizer(pre_model.net, args, pretrain=True)
    pre_scheduler = optim_utils.make_scheduler(pre_optimizer, args, pretrain=True)

    pre_model = DDP(pre_model, device_ids=[gpu], find_unused_parameters=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.rank==0:
        save_results = save_utils.SaveResults(args)
        writer, save_name_linear = save_results.create_results()
        save_results.save_parameters()



    for epoch in range(1, args.epochs+1):
        sampler.set_epoch(epoch)
        # Training 
        # train_loss, train_cov_loss, train_sim_loss, break_code = pretrain_cov(pre_model, pre_train_loader, covariance_loss, pre_optimizer, args.cov_loss_weight, args.sim_loss_weight, epoch, scaler, grad_accumulation_steps=args.grad_accumulation_steps, half=args.half, amp=args.amp)
        # Swap these
        train_loss, train_cov_loss, train_sim_loss, break_code = pretrain_cov_dist(pre_model, pre_train_loader, pre_optimizer, invariance_loss, epoch, scaler, grad_accumulation_steps=args.grad_accumulation_steps, half=args.half, amp=args.amp)
        # train_loss, train_cov_loss, train_sim_loss, break_code = pretrain_cov(pre_model.net, pre_train_loader, covariance_loss, pre_optimizer, args.cov_loss_weight, args.sim_loss_weight, epoch, scaler, grad_accumulation_steps=args.grad_accumulation_steps, half=args.half, amp=args.amp)
        # end swap these
        if break_code:
            test_acc1 = 0.0
            break


        # get learning rate and gradient norm
        pre_curr_lr = pre_optimizer.param_groups[0]['lr']
        total_norm = grad_norm(pre_model)

        # Save results
        if args.rank==0:
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

        # Model and optimizer setup 
        # backbone = copy.deepcopy(pre_model.backbone).cuda()
        # backbone.requires_grad_(False)
        
    if args.rank == 0:
        save_results.save_model_resnet(pre_model)    



if __name__ == '__main__':
    
    # If you would like to remove randomness, uncomment set_seed (extra training time)
    # set_seed(10)
    parser = parsing_file.create_parser()
    arguments = parser.parse_args()
    
    arguments.lin_epochs = arguments.epochs

    # Parameters part for hyperparameter part
    # If you would like to use only command line, uncomment train_test() then you should remove all below
    train_test(arguments)
