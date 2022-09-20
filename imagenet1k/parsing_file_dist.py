import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='jan22 LDMI')
    parser.add_argument('--con_name', default='ldmi', type=str, help='Extra things to define run')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10 or cifar100')
    parser.add_argument('--dataset_location', default='default', type=str, help='Dataset location: default for the default location.')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help="Number of iteration over all dataset to train")
    parser.add_argument('--result_folder', default='results', type=str, help='folder where result files are saved')
    parser.add_argument('--tensor_folder', default='tensorboard', type=str, help='folder where tensorboard files are saved ')

    parser.add_argument('--model_name', default='resnet18', type=str, help='Model name: ResNet18 or ResNet50')
    parser.add_argument('--loss_type', default='covdet', type=str, help='covdet or ldmi')

    # Projector dimensions (In this default setting: IN(512)->L1(2048)->L2(2048)->OUT(128))
    parser.add_argument("--projector", default='512-512-256', type=str, help='projector MLP')

    # Loss weights 
    parser.add_argument("--sim_loss_weight", default=1.0, type=float)
    parser.add_argument("--cov_loss_weight", default=1.0, type=float) 

    # R_initialization and update parameters
    parser.add_argument("--R_ini", default=1.0, type=float)
    parser.add_argument("--la_R", default=0.01, type=float)
    parser.add_argument("--la_mu", default=0.01, type=float)
    
    parser.add_argument("--R_eps_weight", default=1e-6, type=float)
    parser.add_argument("--R_eps_weight2", default=1e-6, type=float)

    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate') 
    parser.add_argument("--normalize_on", action="store_true")
    #parser.add_argument("--scheduler_on", action="store_false")

    parser.add_argument('--pre_optimizer', default='AdamW', type=str, help='Adam, AdamW, SGD')
    parser.add_argument('--ext_opt', default='None', type=str, help='LARS')
    parser.add_argument('--pre_scheduler', default='None', type=str, help='None, lin_warmup_cos, cos, reduce_plateau, multi_step, step')
    #parser.add_argument("--gpu_no", type=int, default=0)

    

    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--warmup_start_lr", default=3e-3, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)

    parser.add_argument("--n_workers", default=8, type=int)

    parser.add_argument("--w_decay", default=1e-6, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)

    parser.add_argument("--eta_lars", default=0.02 , type=float)
    parser.add_argument("--plateau_factor", default=0.95 , type=float)
    parser.add_argument("--plateau_patience", default=10 , type=int)
    parser.add_argument("--plateau_threshold", default=0.01 , type=float)

    parser.add_argument("--step_list", nargs="+", default=[30, 60])
    parser.add_argument("--step_gamma", default=0.1 , type=float)
    parser.add_argument("--step_size", default=30, type=int)

    parser.add_argument("--cov_decay_rate", default=1.0, type=float)
    parser.add_argument("--cos_cov", action="store_true")
    parser.add_argument("--min_cos_cov", default=1.0, type=float)
    parser.add_argument("--exp_decayed", action="store_true") 

    ## Linear
    parser.add_argument('--lin_con_name', default='b', type=str, help='Extra things to define run')
    parser.add_argument('--lin_dataset', default='cifar10', type=str, help='Dataset: cifar10 or tinyimagenet or stl10')
    parser.add_argument('--lin_dataset_location', default='default', type=str, help='Linear dataset location: default for the default location.')
    parser.add_argument('--lin_batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lin_epochs', default=100, type=int, help="Number of iteration over all dataset to train")

    parser.add_argument('--lin_model_name', default='resnet18', type=str, help='Model name: ResNet18 or ResNet50')
    parser.add_argument('--lin_model_path', default='None', type=str, help='Model path: results/xyz.pth')
    
    parser.add_argument('--subset', type=float, default=1.0, help='subset ratio of train dataset')
    parser.add_argument('--subset_type', default="full", help='full, linear, pretrain_and_linear')

    parser.add_argument('--lin_learning_rate', default=0.1, type=float, help='learning rate') 

    parser.add_argument('--lin_optimizer', default='AdamW', type=str, help='Adam, AdamW, SGD')
    parser.add_argument('--lin_scheduler', default='None', type=str, help='None, cos, lin_warmup_cos, reduce_plateau, multi_step, step')

    parser.add_argument("--lin_min_lr", default=1e-6, type=float)
    parser.add_argument("--lin_warmup_start_lr", default=3e-3, type=float)
    parser.add_argument("--lin_warmup_epochs", default=10, type=int)

    parser.add_argument("--lin_w_decay", default=1e-6, type=float)
    parser.add_argument("--lin_momentum", default=0.9, type=float)

    parser.add_argument("--lin_plateau_factor", default=0.95 , type=float)
    parser.add_argument("--lin_plateau_patience", default=10 , type=int)
    parser.add_argument("--lin_plateau_threshold", default=0.01 , type=float)

    parser.add_argument("--lin_step_list", nargs="+", default=[30, 60])
    parser.add_argument("--lin_step_gamma", default=0.1 , type=float)
    parser.add_argument("--lin_step_size", default=30, type=int)

    # Optimization 
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--lin_grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--amp", action="store_true") 
    parser.add_argument("--half", action="store_true") 

    # Distributed
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int) # default=-1,
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    # Many more arguments
    return parser