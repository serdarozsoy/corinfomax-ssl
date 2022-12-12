import torch
from torch.utils.data import DataLoader

from loss import CovarianceLoss

from model_base_resnet import CovModel, LinModel
from metrics import correct_top_k, grad_norm

from pretrain_base import pretrain_cov
from linear_base import linear_train, linear_test
from lin_classifier import LinClassifier
import parsing_file
import data_utils
import optim_utils
import save_utils_con as save_utils

import copy
import numpy as np
import random
import os



def train_test(args):
    """
    Pretrain for args.epochs, then linear evaluation (train and test) for args.lin_epochs. 
    """
    pretrain_data, lin_train_data, lin_test_data  = data_utils.make_data(args.dataset, args.subset, args.subset_type)
    
    pre_train_loader = DataLoader(pretrain_data, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True, drop_last=True) 
    lin_train_loader = DataLoader(lin_train_data, batch_size=args.lin_batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)
    lin_test_loader = DataLoader(lin_test_data, batch_size=args.lin_batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True)


    # Initialize corinfomax loss 
    covariance_loss = CovarianceLoss(args)


    # Model and optimizer setup 
    pre_model = CovModel(args).cuda()
    pre_optimizer = optim_utils.make_optimizer(pre_model, args, pretrain=True)
    pre_scheduler = optim_utils.make_scheduler(pre_optimizer, args, pretrain=True)

    save_results = save_utils.SaveResults(args)
    writer = save_results.create_results()
    save_results.save_parameters()


    for epoch in range(1, args.epochs+1):
        # Pretraining 
        train_loss, train_cov_loss, train_sim_loss, break_code = pretrain_cov(pre_model, pre_train_loader, covariance_loss, pre_optimizer, args.cov_loss_weight, args.sim_loss_weight, epoch)
        if break_code:
            test_acc1 = 0.0
            break
        # Scheduler action
        if args.pre_scheduler == "None":
            pass
        elif args.pre_scheduler == "reduce_plateau":
            pre_scheduler.step(train_loss)
        else:
            pre_scheduler.step()
        # get learning rate from optimizer
        pre_curr_lr = pre_optimizer.param_groups[0]['lr']
        total_norm = grad_norm(pre_model)

        save_results.update_results(train_loss, train_cov_loss, train_sim_loss, pre_curr_lr, total_norm)
        R_eigs = save_results.save_eigs(covariance_loss, epoch)

        save_utils.update_tensorboard(writer, train_loss, train_cov_loss, train_sim_loss, pre_curr_lr, total_norm, R_eigs, epoch)
        save_results.save_stats(epoch)

        writer.flush()

        linear_period = args.epochs

        if epoch % 200 == 0:
            save_results.save_model(pre_model, pre_optimizer, train_loss, epoch)

        if epoch % linear_period == 0:
            save_results.save_model(pre_model, pre_optimizer, train_loss, epoch)

            # Initialize linear classifier 
            lin_classifier = LinClassifier(args, offline=True).cuda()
            # Model and optimizer setup 
            backbone = pre_model.backbone
            backbone.requires_grad_(False)

            lin_optimizer = optim_utils.make_optimizer(lin_classifier, args, pretrain=False)
            lin_scheduler = optim_utils.make_scheduler(lin_optimizer, args, pretrain=False)


            for lin_epoch in range(1, args.lin_epochs+1):
                # Training for linear evaluation 
                linear_loss, linear_acc1, linear_acc5 = linear_train(backbone, lin_train_loader, lin_optimizer, lin_classifier, lin_epoch)

                # Scheduler action
                if args.lin_scheduler == "None":
                    pass
                elif args.lin_scheduler == "reduce_plateau":
                    lin_scheduler.step(linear_loss)
                else:
                    lin_scheduler.step()
                # get learning rate from optimizer
                lin_curr_lr = lin_optimizer.param_groups[0]['lr']
                total_lin_norm = grad_norm(lin_classifier)
                with torch.no_grad():
                    # Testing for linear evaluation
                    test_loss, test_acc1, test_acc5 = linear_test(backbone, lin_test_loader, lin_classifier, lin_epoch)

                save_epoch = lin_epoch
                save_results.update_lin_results(linear_loss, linear_acc1, linear_acc5, lin_curr_lr, test_loss, test_acc1, test_acc5, total_lin_norm)
                save_utils.update_lin_tensorboard(writer, linear_loss, linear_acc1, linear_acc5, lin_curr_lr, test_loss, test_acc1, test_acc5, total_lin_norm, save_epoch)
                save_results.save_lin_stats(save_epoch,)

                writer.flush()



if __name__ == '__main__':

    parser = parsing_file.create_parser()
    arguments = parser.parse_args()
    
    train_test(arguments)
