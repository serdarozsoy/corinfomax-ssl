import torch
from tqdm import tqdm
from metrics import correct_top_k
import torch.nn.functional as F

def linear_test(net, data_loader, epoch):
    # evaluate model:
    net.eval()
    linear_loss = 0.0
    num = 0
    total_loss, total_correct_1, total_correct_5, total_num, test_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with torch.no_grad():
        for data_tuple in test_bar:
            data, target = [t.cuda() for t in data_tuple]

            # Forward prop with only f (ResNet) part of model
            output= net(data) #net.f(data)
            # Calculate Loss for batch
            linear_loss = F.cross_entropy(output, target)
            
            # Batchsize for loss and accuracy
            num = data.size(0) * get_world_size()
            total_num += num 
            
            # Accumulating loss 
            total_loss += linear_loss.item() * num #data[0].size(0) #data.size(0)  #pos_1.size(0)

            # Accumulating number of correct predictions 
            correct_top_1, correct_top_5 = correct_top_k(output, target, top_k=(1,5))    
            total_correct_1 += correct_top_1
            total_correct_5 += correct_top_5

            test_bar.set_description('Lin.Test Epoch: [{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}% '
                                     .format(epoch,  total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100
                                             ))
        acc_1 = total_correct_1/total_num*100
        acc_5 = total_correct_5/total_num*100
    return total_loss / total_num, acc_1 , acc_5 



def linear_train(net, data_loader, train_optimizer, epoch):
    net.eval() # "eval" for not update any batchnorm but "train" for tuning
    total_num, train_bar = 0, tqdm(data_loader)
    linear_loss = 0.0
    total_correct_1, total_correct_5 = 0.0, 0.0
    for data_tuple in train_bar:
        # Forward prop of the model for both inputs
        pos_1, target = data_tuple
        pos_1 = pos_1.cuda()

        # Batchsize before concat
        batchsize_bc = pos_1.shape[0]
        targets = target.cuda()


        # logits = net(pos_1) 
        # # Classifier with detach(for stop gradient to model)
        # #logits = classifier(features.detach()) #classifier.forward(features.detach())
        # # Loss 
        # linear_loss_1 = F.cross_entropy(logits, targets)

        # # Number of correct predictions
        # linear_correct_1, linear_correct_5 = correct_top_k(logits, targets, top_k=(1, 5))
        linear_loss_1, linear_correct_1, linear_correct_5 = net(pos_1, targets=targets)
    
        # Batchsize after concat
        # batchsize_ac = features.shape[0]

        # Backpropagation part
        train_optimizer.zero_grad()
        linear_loss_1.backward()
        train_optimizer.step()

        # Accumulating number of examples, losses and correct predictions
        total_num += batchsize_bc
        linear_loss += linear_loss_1.item() * batchsize_bc
        total_correct_1 += linear_correct_1 
        total_correct_5 += linear_correct_5

        # This bar is used for live tracking on command line (batch_size -> batchsize_bc: to show current batchsize )
        train_bar.set_description('Lin.Train Epoch: [{}] Loss: {:.4f} '.format(\
                epoch, linear_loss / total_num))
    
    acc_1 = total_correct_1/total_num*100
    acc_5 = total_correct_5/total_num*100
    
    return linear_loss/total_num, acc_1, acc_5
