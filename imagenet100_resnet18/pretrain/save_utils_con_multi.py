import pandas as pd
import os 
import time
import json
import torch
from torch.utils.tensorboard import SummaryWriter


class SaveResults:
    def __init__(self, args): #, result_folder="results", tensor_folder="tensorboard"
        super().__init__()
        self.args = args
        self.result_folder=args.result_folder #result_folder
        self.tensor_folder=args.tensor_folder #tensor_folder
        self.save_name_pre = "empty"
        self.parameters_dict = {}
        self.results = {}
        self.lin_results = {}
        


    def create_results(self):
        args = self.args
        self.results = {'train_loss': [],  
                   'train_cov_loss': [], 
                   'train_sim_loss': [], 
                   'learning_rate': [], 
                   'train_grad_norm': []}
        self.lin_results = { 'linear_lr': [], 
                            'linear_loss':[], 
                            'linear_acc1':[],
                            'linear_acc5':[], 
                            'test_loss':[], 
                            'test_acc1':[],
                            'test_acc5':[],
                            'lin_grad_norm':[]}
    
        unique_time = int(time.time()) 
        self.save_name_pre = 't_{}_{}'.format(unique_time, args.con_name)


        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        if not os.path.exists(self.tensor_folder):
            os.mkdir(self.tensor_folder)

        # Tensorboard
        dir_path = os.path.dirname(os.path.realpath(__file__))
        tensorboard_save_path = '{}/{}/{}.pth'.format(dir_path, self.tensor_folder, self.save_name_pre)
        writer = SummaryWriter(log_dir=tensorboard_save_path)

        return writer, self.save_name_pre


    def save_parameters(self):
        args = self.args
        self.parameters_dict = {"dataset" : args.dataset,
        "projector": args.projector,
        "batch_size":  args.batch_size,
        "epochs" : args.epochs,
        "learning_rate" : args.learning_rate,
        "model_name" : args.model_name,
        "sim_loss_weight" : args.sim_loss_weight,
        "cov_loss_weight" : args.cov_loss_weight,
        "R_ini" : args.R_ini,
        "la_R" : args.la_R,
        "la_mu" : args.la_mu,
        "normalize_on" : args.normalize_on,
        "R_eps_weight" : args.R_eps_weight,
        "pre_optimizer" : args.pre_optimizer,
        "pre_scheduler" : args.pre_scheduler,
        "ext_opt" : args.ext_opt,
        "min_lr" : args.min_lr,
        "warmup_start_lr" : args.warmup_start_lr,
        "warmup_epochs" : args.warmup_epochs,
        "n_workers" : args.n_workers,
        "w_decay" : args.w_decay,
        "momentum" : args.momentum,
        "eta_lars" : args.eta_lars,
        "plateau_factor" : args.plateau_factor,
        "plateau_patience" : args.plateau_patience,
        "plateau_threshold" : args.plateau_threshold,
        "lin_batch_size":  args.lin_batch_size,
        "lin_epochs" : args.lin_epochs,
        "lin_learning_rate" : args.lin_learning_rate,
        "lin_optim_name" : args.lin_optimizer,
        "lin_sched_name" : args.lin_scheduler,
        "lin_min_lr" : args.lin_min_lr,
        "lin_warmup_start_lr" : args.lin_warmup_start_lr,
        "lin_warmup_epochs" : args.lin_warmup_epochs,
        "lin_w_decay" : args.lin_w_decay,
        "lin_momentum" : args.lin_momentum,
        "lin_plateau_factor" : args.lin_plateau_factor,
        "lin_plateau_patience" : args.lin_plateau_patience,
        "lin_plateau_threshold" : args.lin_plateau_threshold,
        "subset" : args.subset, 
        "subset_type": args.subset_type,
        "cov_decay_rate": args.cov_decay_rate,
        "cos_cov": args.cos_cov,
        "min_cos_cov": args.min_cos_cov,
        "exp_decayed":args.exp_decayed
        }

        json_path = '{}/{}_params.json'.format(self.result_folder, self.save_name_pre)
        with open(json_path, 'w') as f:
            json.dump(self.parameters_dict, f, indent=4)


    def save_eigs(self, loss, epoch):
        R_eigs = loss.save_eigs() #, Re_eigs
        R_eigs_df = pd.DataFrame(data=R_eigs, index=range(0, epoch+1))
        R_eigs_df.to_csv('{}/{}_R_eigs.csv'.format(self.result_folder, self.save_name_pre), index_label='epoch')
        return R_eigs

    def save_stats(self, epoch):
        # Save list as csv
        data_frame = pd.DataFrame(data=self.results, index=range(1, epoch+1, 1))
        data_frame.to_csv('{}/{}_statistics.csv'.format(self.result_folder, self.save_name_pre), index_label='epoch')

    def save_lin_stats(self, epoch):
        # Save list as csv
        data_frame = pd.DataFrame(data=self.lin_results, index=range(1, epoch+1, 1))
        data_frame.to_csv('{}/{}_lin_statistics.csv'.format(self.result_folder, self.save_name_pre), index_label='epoch')



    def save_model(self, model, optimizer, train_loss, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'parameters_dict': self.parameters_dict,
            'loss': train_loss,
            #'R': covariance_loss.R,
            #'mu': covariance_loss.mu,
            }, '{}/{}_model{}.pth'.format(self.result_folder, self.save_name_pre, epoch))


    def save_model_resnet(self, model):
        torch.save(
            model.module.net.backbone.state_dict(), '{}/{}_resnet50.pth'.format(self.result_folder, self.save_name_pre))


    def update_results(self, train_loss, train_cov_loss, train_sim_loss, pre_curr_lr, total_norm):
        self.results['train_loss'].append(train_loss)
        self.results['train_cov_loss'].append(train_cov_loss)
        self.results['train_sim_loss'].append(train_sim_loss)
        self.results['learning_rate'].append(pre_curr_lr)
        self.results['train_grad_norm'].append(total_norm)


    def update_lin_results(self, linear_loss, linear_acc1, linear_acc5, lin_curr_lr, test_loss, test_acc1, test_acc5, total_norm):
        self.lin_results['linear_lr'].append(lin_curr_lr)
        self.lin_results['linear_loss'].append(linear_loss)
        self.lin_results['linear_acc1'].append(linear_acc1)
        self.lin_results['linear_acc5'].append(linear_acc5)
        self.lin_results['test_loss'].append(test_loss)
        self.lin_results['test_acc1'].append(test_acc1)
        self.lin_results['test_acc5'].append(test_acc5)
        self.lin_results['lin_grad_norm'].append(total_norm)


def update_tensorboard(writer, train_loss, train_cov_loss, train_sim_loss, pre_curr_lr, total_norm, R_eigs, epoch):
    writer.add_scalar("Train/Loss", train_loss, epoch)
    writer.add_scalar("Train/Cov_Loss", train_cov_loss, epoch)
    writer.add_scalar("Train/Sim_Loss", train_sim_loss, epoch)
    writer.add_scalar("Train/Learning_Rate", pre_curr_lr , epoch)
    writer.add_scalar("Train/low_R_eig", (R_eigs[epoch,:] < 1e-8).sum(), epoch)
    writer.add_scalar("Train/high_R_eig", (R_eigs[epoch,:] > 1e-1).sum(), epoch)
    writer.add_scalar("Train/grad_norm", total_norm, epoch)

def update_lin_tensorboard(writer, linear_loss, linear_acc1, linear_acc5, lin_curr_lr, test_loss, test_acc1, test_acc5, total_norm, epoch):
    writer.add_scalar("Lin_Train/Loss", linear_loss, epoch)
    writer.add_scalar("Lin_Train/Acc1", linear_acc1, epoch)
    writer.add_scalar("Lin_Train/Acc5", linear_acc5, epoch)
    writer.add_scalar("Lin_Train/Learning_Rate", lin_curr_lr , epoch)
    writer.add_scalar("Lin_Train/grad_norm", total_norm, epoch)
    writer.add_scalar("Lin_Test/Loss", test_loss, epoch)
    writer.add_scalar("Lin_Test/Acc1", test_acc1, epoch)
    writer.add_scalar("Lin_Test/Acc5", test_acc5, epoch)