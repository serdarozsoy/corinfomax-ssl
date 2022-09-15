import torch
from tqdm import tqdm
from metrics import correct_top_k
import torch.nn.functional as F

def linear_test(net, data_loader, classifier, epoch):
    # evaluate model:
    net.eval() # for not update batchnorm
    linear_loss = 0.0
    num = 0
    total_loss, total_correct_1, total_correct_5, total_num, test_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with torch.no_grad():
        for data_tuple in test_bar:
            data, target = [t.cuda() for t in data_tuple]

            # Forward prop of the model with single augmented batch
            feature = net(data) 

            # Logits by classifier
            output = classifier(feature) 

            # Calculate Cross Entropy Loss for batch
            linear_loss = F.cross_entropy(output, target)
            
            # Batchsize for loss and accuracy
            num = data.size(0)
            total_num += num 
            
            # Accumulating loss 
            total_loss += linear_loss.item() * num 
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



def linear_train(net, data_loader, train_optimizer, classifier, epoch):
    net.eval() # for not update batchnorm 
    total_num, train_bar = 0, tqdm(data_loader)
    linear_loss = 0.0
    total_correct_1, total_correct_5 = 0.0, 0.0
    for data_tuple in train_bar:
        # Forward prop of the model with single augmented batch
        pos_1, target = data_tuple
        pos_1 = pos_1.cuda()
        feature_1 = net(pos_1) 

        # Batchsize
        batchsize_bc = feature_1.shape[0]
        features = feature_1
        targets = target.cuda()


        # Classifier with detach(for stop gradient to model)
        # Actually no need due to we use lin_optimizer seperate
        logits = classifier(features.detach()) 
        # Cross Entropy Loss 
        linear_loss_1 = F.cross_entropy(logits, targets)

        # Number of correct predictions
        linear_correct_1, linear_correct_5 = correct_top_k(logits, targets, top_k=(1, 5))
    

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
