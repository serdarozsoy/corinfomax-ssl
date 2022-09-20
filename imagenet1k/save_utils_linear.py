import pandas as pd
import os 
import time
import json
import torch
from torch.utils.tensorboard import SummaryWriter


class SaveResults:
    def __init__(self, args, result_folder="results", tensor_folder="tensorboard"):
        super().__init__()
        self.args = args
        self.result_folder=result_folder 
        self.tensor_folder=tensor_folder
        self.save_name_pre = "empty"
        self.parameters_dict = {}
        self.lin_results = {}

        


    def create_results(self):
        args = self.args
        self.lin_results = { 'linear_lr': [], 
                            'linear_loss':[], 
                            'linear_acc1':[],
                            'linear_acc5':[], 
                            'test_loss':[], 
                            'test_acc1':[],
                            'test_acc5':[],
                            'lin_grad_norm':[]}
    
        # unique_time = int(time.time()) 
        # self.save_name_pre = 't_{}_{}'.format(unique_time, args.con_name)
        
        lin_model_path = args.lin_model_path
        save_name_pre_tensorboard = lin_model_path.split("/")[-1].split("_model")[0]
        model_epoch = lin_model_path .split("_model")[-1].split("_")[0].split(".")[0]

        unique_time = int(time.time()) 
        self.save_name_pre = '{}_{}_ep{}_{}_linear'.format(save_name_pre_tensorboard, args.con_name, model_epoch, unique_time)



        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        if not os.path.exists(self.tensor_folder):
            os.mkdir(self.tensor_folder)

        # Tensorboard
        dir_path = os.path.dirname(os.path.realpath(__file__))
        tensorboard_save_path = '{}/{}/{}.pth'.format(dir_path, self.tensor_folder, self.save_name_pre)
        writer = SummaryWriter(log_dir=tensorboard_save_path)


        return writer


    def save_parameters(self):
        args = self.args
        self.parameters_dict = {"dataset" : args.dataset,
        "lin_model_path" : args.lin_model_path,
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
        }

        json_path = '{}/{}_params.json'.format(self.result_folder, self.save_name_pre)
        with open(json_path, 'w') as f:
            json.dump(self.parameters_dict, f, indent=4)

    def save_lin_stats(self, epoch):
        # Save list as csv
        data_frame = pd.DataFrame(data=self.lin_results, index=range(1, epoch+1, 1))
        data_frame.to_csv('{}/{}_lin_statistics.csv'.format(self.result_folder, self.save_name_pre), index_label='epoch')



    def save_model(self, model, optimizer, train_loss, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'parameters_dict': self.parameters_dict,
            'loss': train_loss,
            #'R': covariance_loss.R,
            #'mu': covariance_loss.mu,
            }, '{}/{}_model{}_linear.pth'.format(self.result_folder, self.save_name_pre, epoch))


    def update_lin_results(self, linear_loss, linear_acc1, linear_acc5, lin_curr_lr, test_loss, test_acc1, test_acc5, total_norm):
        self.lin_results['linear_lr'].append(lin_curr_lr)
        self.lin_results['linear_loss'].append(linear_loss)
        self.lin_results['linear_acc1'].append(linear_acc1)
        self.lin_results['linear_acc5'].append(linear_acc5)
        self.lin_results['test_loss'].append(test_loss)
        self.lin_results['test_acc1'].append(test_acc1)
        self.lin_results['test_acc5'].append(test_acc5)
        self.lin_results['lin_grad_norm'].append(total_norm)


def update_lin_tensorboard(writer, linear_loss, linear_acc1, linear_acc5, lin_curr_lr, test_loss, test_acc1, test_acc5, total_norm, epoch):
    writer.add_scalar("Lin_Train/Loss", linear_loss, epoch)
    writer.add_scalar("Lin_Train/Acc1", linear_acc1, epoch)
    writer.add_scalar("Lin_Train/Acc5", linear_acc5, epoch)
    writer.add_scalar("Lin_Train/Learning_Rate", lin_curr_lr , epoch)
    writer.add_scalar("Lin_Train/grad_norm", total_norm, epoch)
    writer.add_scalar("Lin_Test/Loss", test_loss, epoch)
    writer.add_scalar("Lin_Test/Acc1", test_acc1, epoch)
    writer.add_scalar("Lin_Test/Acc5", test_acc5, epoch)