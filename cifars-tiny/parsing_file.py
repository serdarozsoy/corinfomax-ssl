import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='corinfomax')
    parser.add_argument('--con_name', default='ldmi', type=str, help='Extra things to define run')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10, cifar100, tiny-imagenet')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help="Number of iteration over all dataset to train")
    parser.add_argument('--result_folder', default='results', type=str, help='folder where result files are saved')
    parser.add_argument('--tensor_folder', default='tensorboard', type=str, help='folder where tensorboard files are saved ')

    parser.add_argument('--model_name', default='resnet18', type=str, help='encoder network: resnet18 or resnet50')
    parser.add_argument('--loss_type', default='corinfomax', type=str, help='name of the loss')

    # Projector dimensions (In this default setting: IN(512)->L1(2048)->L2(2048)->OUT(64))
    parser.add_argument("--projector", default='2048-2048-64', type=str, help='projector MLP')

    # Loss weights 
    parser.add_argument("--sim_loss_weight", default=500.0, type=float, help='alpha coefficient')
    parser.add_argument("--cov_loss_weight", default=1.0, type=float, help='constant=1') 

    # R_initialization and update parameters
    parser.add_argument("--R_ini", default=1.0, type=float, help='coefficient of initial covariance (identity matrix)')
    parser.add_argument("--la_R", default=0.01, type=float, help='forgetting factor for covariance matrix')
    parser.add_argument("--la_mu", default=0.01, type=float, help='forgetting factor for projector output mean')
    
    parser.add_argument("--R_eps_weight", default=1e-6, type=float, help='diagonal perturbation factor of covariance matrix R1')
    parser.add_argument("--R_eps_weight2", default=1e-6, type=float, help='diagonal perturbation factor of covariance matrix R2')

    parser.add_argument('--learning_rate', default=0.5, type=float, help='learning rate')
    parser.add_argument("--normalize_on", action="store_true", help='l2 normalization after projection MLP')

    parser.add_argument('--pre_optimizer', default='SGD', type=str, help='Adam, AdamW, SGD')
    parser.add_argument('--ext_opt', default='None', type=str, help='LARS')
    parser.add_argument('--pre_scheduler', default='lin_warmup_cos', type=str, help='None, lin_warmup_cos, cos, reduce_plateau, multi_step, step')
    parser.add_argument("--gpu_no", type=int, default=0)

    parser.add_argument("--min_lr", default=5e-3, type=float, help='minimum learning rate for decaying schedulers')
    parser.add_argument("--warmup_start_lr", default=3e-3, type=float, help='initial learning rate for warmup period')
    parser.add_argument("--warmup_epochs", default=10, type=int, help='number of epochs for warmup period' )

    parser.add_argument("--n_workers", default=8, type=int, help='for multiprocess data loading')

    parser.add_argument("--w_decay", default=1e-4, type=float, help='weight decay for pretraining')
    parser.add_argument("--momentum", default=0.9, type=float, help='momentum factor' )

    parser.add_argument("--eta_lars", default=0.02 , type=float, help='LARS parameter')
    parser.add_argument("--plateau_factor", default=0.95 , type=float, help='multiplication factor of learning rate in reduce_plateau scheduler')
    parser.add_argument("--plateau_patience", default=10 , type=int, help='waiting epoch number for new update in reduce_plateau scheduler')
    parser.add_argument("--plateau_threshold", default=0.01 , type=float, help='threshold rate in reduce_plateau scheduler')

    parser.add_argument("--step_list", nargs="+", default=[30, 60], help='step list for multi step scheduler')
    parser.add_argument("--step_gamma", default=0.1 , type=float, help='multiplication factor for step scheduler')
    parser.add_argument("--step_size", default=30, type=int, help='step size for step scheduler')

    parser.add_argument("--cov_decay_rate", default=1.0, type=float, help='experimental')
    parser.add_argument("--cos_cov", action="store_true", help='experimental')
    parser.add_argument("--min_cos_cov", default=1.0, type=float, help='experimental')
    parser.add_argument("--exp_decayed", action="store_true", help='experimental') 

    ## Linear evaluation parameters
    parser.add_argument('--lin_con_name', default='b', type=str, help='Extra things to define run')
    parser.add_argument('--lin_dataset', default='cifar10', type=str, help='Dataset: cifar10, cifar100 or tinyimagenet')
    parser.add_argument('--lin_batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lin_epochs', default=100, type=int, help="Number of iteration over all dataset to train in linear evaluation")

    parser.add_argument('--lin_model_name', default='resnet18', type=str, help='encoder network: resnet18 or resnet50')
    parser.add_argument('--lin_model_path', default='None', type=str, help='pretrained model path: results/xyz.pth')
    
    parser.add_argument('--subset', type=float, default=1.0, help='subset ratio of train dataset')
    parser.add_argument('--subset_type', default="full", help='full, linear, pretrain_and_linear')

    parser.add_argument('--lin_learning_rate', default=0.1, type=float, help='learning rate in linear evaluation') 

    parser.add_argument('--lin_optimizer', default='SGD', type=str, help='Adam, AdamW, SGD')
    parser.add_argument('--lin_scheduler', default='cos', type=str, help='None, cos, lin_warmup_cos, reduce_plateau, multi_step, step')

    parser.add_argument("--lin_min_lr", default=2e-3, type=float, help='minimum learning rate for decaying schedulers in linear evaluation')
    parser.add_argument("--lin_warmup_start_lr", default=3e-3, type=float, help='initial learning rate for warmup period in linear evaluation')
    parser.add_argument("--lin_warmup_epochs", default=10, type=int, help='number of epochs for warmup period in linear evaluation')

    parser.add_argument("--lin_w_decay", default=0, type=float, help='weight decay for pretraining in linear evaluation')
    parser.add_argument("--lin_momentum", default=0.9, type=float, help='momentum factor in linear evaluation')

    parser.add_argument("--lin_plateau_factor", default=0.95 , type=float, help='multiplication factor of learning rate in reduce_plateau scheduler')
    parser.add_argument("--lin_plateau_patience", default=10 , type=int, help='waiting epoch number for new update in reduce_plateau scheduler')
    parser.add_argument("--lin_plateau_threshold", default=0.01 , type=float, help='threshold rate in reduce_plateau scheduler')

    parser.add_argument("--lin_step_list", nargs="+", default=[30, 60], help='step list for multi step scheduler')
    parser.add_argument("--lin_step_gamma", default=0.1 , type=float,  help='multiplication factor for step scheduler')
    parser.add_argument("--lin_step_size", default=30, type=int, help='step size for step scheduler')
    # Many more arguments can be added
    return parser