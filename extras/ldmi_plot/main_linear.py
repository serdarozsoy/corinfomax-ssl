import torch
from torch.utils.data import DataLoader

from loss import CovarianceLoss

from model_base_resnet import LinModel
from metrics import correct_top_k, grad_norm

from linear_base import linear_train, linear_test
from lin_classifier import LinClassifier
import parsing_file
import data_utils
import optim_utils
import save_utils_linear as save_utils


def train_test(args):
    """
    Linear evaluation (train and test) for args.lin_epochs.
    """
    pretrain_data, lin_train_data, lin_test_data  = data_utils.make_data(args.dataset, args.subset, args.subset_type)
    
    
    lin_train_loader = DataLoader(lin_train_data, batch_size=args.lin_batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)
    lin_test_loader = DataLoader(lin_test_data, batch_size=args.lin_batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True)

    # Initialize linear classifier 
    lin_classifier = LinClassifier(args, offline=True).cuda()

    # Model setup for linear evaluation
    backbone = LinModel(args).cuda()

    # Load pretrained model and disable gradient calculation
    checkpoint = torch.load(args.lin_model_path,  map_location="cpu") 
    backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
    backbone.requires_grad_(False)

    # Optimizer and scheduler setup
    lin_optimizer = optim_utils.make_optimizer(lin_classifier, args, pretrain=False)
    lin_scheduler = optim_utils.make_scheduler(lin_optimizer, args, pretrain=False)

    save_results = save_utils.SaveResults(args)
    writer = save_results.create_results()
    save_results.save_parameters()

    # Loop for training and test together in linear evaluation
    for lin_epoch in range(1, args.lin_epochs+1):
        # Linear evaluation - Training 
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
            # Linear evaluation - Test
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