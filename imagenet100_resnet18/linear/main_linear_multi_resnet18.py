import torch
from torch.utils.data import DataLoader

from loss import CovarianceLoss

from model_base_wmse_resnet18  import LinModel
from metrics import correct_top_k, grad_norm

from linear_base_dist import linear_train, linear_test
from lin_classifier import LinClassifier
import parsing_file_dist as parsing_file
import data_utils_dist as data_utils
import optim_utils
import save_utils_linear as save_utils

import numpy as np
import random
import os

from distributed import init_distributed_mode
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def test(args):


    #set_seed(10)
    #torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    gpu = torch.device(args.device)

    _ , lin_train_data, lin_test_data  = data_utils.make_data(args.dataset, args.subset, args.subset_type)


    sampler = torch.utils.data.distributed.DistributedSampler(lin_train_data, shuffle=True)
    assert args.lin_batch_size % args.world_size == 0
    per_device_batch_size = args.lin_batch_size // args.world_size

    lin_train_loader = DataLoader(lin_train_data, batch_size=per_device_batch_size, num_workers=args.n_workers, sampler=sampler, pin_memory=True, drop_last=True)  

    lin_test_loader = DataLoader(lin_test_data, batch_size=args.lin_batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True)

    # Initialize loss and classifier 
    lin_classifier = LinClassifier(args, offline=True).cuda(gpu)
    # Model and optimizer setup 
    backbone = LinModel(args).cuda(gpu)
    checkpoint = torch.load(args.lin_model_path,  map_location="cpu") 
    backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
    backbone.requires_grad_(False)

    lin_optimizer = optim_utils.make_optimizer(lin_classifier, args, pretrain=False)
    lin_scheduler = optim_utils.make_scheduler(lin_optimizer, args, pretrain=False)

    model = torch.nn.Sequential(backbone, lin_classifier)
    model.cuda(gpu)
    backbone.requires_grad_(False)
    lin_classifier.requires_grad_(True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if args.rank == 0:
        save_results = save_utils.SaveResults(args)
        writer = save_results.create_results()
        save_results.save_parameters()

    # Loop for training and test together
    for lin_epoch in range(1, args.lin_epochs+1):
        sampler.set_epoch(lin_epoch)
        # Training 
        linear_loss, linear_acc1, linear_acc5 = linear_train(model, lin_train_loader, lin_optimizer,lin_epoch)

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
            # Testing
            test_loss, test_acc1, test_acc5 = linear_test(model, lin_test_loader, lin_epoch)

        save_epoch = lin_epoch

        if args.rank == 0:
            save_results.update_lin_results(linear_loss, linear_acc1, linear_acc5, lin_curr_lr, test_loss, test_acc1, test_acc5, total_lin_norm)
            save_utils.update_lin_tensorboard(writer, linear_loss, linear_acc1, linear_acc5, lin_curr_lr, test_loss, test_acc1, test_acc5, total_lin_norm, save_epoch)
            save_results.save_lin_stats(save_epoch)

            writer.flush()

        #if lin_epoch % 10 == 0:
        #if ((lin_epoch == args.lin_epochs)) & (args.rank == 0):
        #    save_results.save_model(backbone, lin_optimizer, linear_loss, lin_epoch)




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

    """
    # Optional for hyperparameter tuning
    
    learning_rate_list = [0.007] 
    batch_size_list = [256]

    for lr_1 in learning_rate_list:
        for bsz_1 in batch_size_list:
            arguments.lin_learning_rate = lr_1
            arguments.lin_batch_size = bsz_1
    """

    test(arguments)