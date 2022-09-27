import torch
import time
import data
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import aux_funcs as af


from data import *
from tqdm import tqdm
from model import *
from util.bce_acc import *
from sklearn import metrics
# from fairness_metric import *
from torchvision import models
from datetime import datetime

def accuracy_metrices(label_list, y_pred_list, sensitive_group_list, sensitive_group_name_list):
    # sensitive_group_name_list[0] : sensitive groups name e.g. gender, races etcs.
    results = {
            '{}_acc'.format(sensitive_group_name_list[0]): metrics.accuracy_score(label_list[sensitive_group_list==0], y_pred_list[sensitive_group_list==0]),
            '{}_acc'.format(sensitive_group_name_list[1]): metrics.accuracy_score(label_list[sensitive_group_list==1], y_pred_list[sensitive_group_list==1]),
            '{}_precision'.format(sensitive_group_name_list[0]): metrics.precision_score(label_list[sensitive_group_list==0], y_pred_list[sensitive_group_list==0], average='macro', zero_division=0),
            '{}_precision'.format(sensitive_group_name_list[1]): metrics.precision_score(label_list[sensitive_group_list==1], y_pred_list[sensitive_group_list==1], average='macro', zero_division=0),
            '{}_recall'.format(sensitive_group_name_list[0]): metrics.recall_score(label_list[sensitive_group_list==0], y_pred_list[sensitive_group_list==0], average='macro', zero_division=0),
            '{}_recall'.format(sensitive_group_name_list[1]): metrics.recall_score(label_list[sensitive_group_list==1], y_pred_list[sensitive_group_list==1], average='macro', zero_division=0),
            '{}_f1_score'.format(sensitive_group_name_list[0]): metrics.f1_score(label_list[sensitive_group_list==0], y_pred_list[sensitive_group_list==0], average='macro', zero_division=0),
            '{}_f1_score'.format(sensitive_group_name_list[1]): metrics.f1_score(label_list[sensitive_group_list==1], y_pred_list[sensitive_group_list==1], average='macro', zero_division=0),
            'accuracy': metrics.accuracy_score(label_list, y_pred_list),
            'F1': metrics.f1_score(label_list, y_pred_list, average='macro'),
    }
    return results

def val_accuracy(output, ta, sa, ta_cls, sa_cls, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = ta.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(ta.view(1, -1).expand_as(pred))

        group=[]
        group_num=[]
        for i in range(ta_cls):
            sa_group=[]
            sa_group_num=[]
            for j in range(sa_cls):
                eps=1e-8
                sa_group.append(((sa==j)*(ta==i)*(correct==1)).float().sum() *(100 /(((sa==j)*(ta==i)).float().sum()+eps)))
                sa_group_num.append(((sa==j)*(ta==i)).float().sum()+eps)
            group.append(sa_group)
            group_num.append(sa_group_num)
       
        res=(correct==1).float().sum()*(100.0 / batch_size)
        
        return res,group,group_num

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cnn_test_fairness(model, loader, device='cpu', model_name='', dataset_name='', eval_final=False):
    print(dataset_name)
    model.eval()
    top1 = AverageMeter()
    groupAcc=[]
    for i in range(2):
        saGroupAcc=[]
        for j in range(2):
            saGroupAcc.append(AverageMeter())
        groupAcc.append(saGroupAcc)
    with torch.no_grad():
        for batch in tqdm(loader):
            b_x = batch[0].cuda()
            b_y = batch[1].cuda()
            gender = batch[2].cuda()
            bsz = b_y.shape[0]
            output = model(b_x)
            pred = None
            _, pred = torch.max(output, 1)
            
            acc1, group_acc, group_num = val_accuracy(output, b_y, gender, 2, 2)
            top1.update(acc1, bsz)

            for i in range(2):
                for j in range(2):
                    groupAcc[i][j].update(group_acc[i][j],group_num[i][j])
              
    odds=0
    odds_num=0
    for i in range(2):
        for j in range(2):
            for k in range(j+1,2):
                odds_num+=1
                odds+=torch.abs(groupAcc[i][j].avg-groupAcc[i][k].avg)
    
    print('\n * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Equalized Odds {odds:.3f}'.format(odds=(odds/odds_num).item()))
    print(' * Group-wise accuracy')
    for i in range(2):
        string='    Target class '+str(i)+'\n    '
        for j in range(2):
            string+= '    Sensitive class '+str(j)+': {groupAcc.avg:.3f}'.format(groupAcc=groupAcc[i][j])
        print(string+'\n') 

def cnn_train(model, data, epochs, optimizer, scheduler, device='cuda', tensor_board_path='', models_path='', sensitive_group_name_list=[]):
    for epoch in range(1, epochs):
        acc_avg = AverageMeter()
        loss = [] 
        label_list = []
        y_pred_list = []
        sensitive_group_list = []
        cur_lr = af.get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.aug_train_loader
        else:
            train_loader = data.train_loader

        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        for x, y, sensitive_group in train_loader:
            b_x = x.to(device)   # batch x
            b_y = y.to(device)   # batch y
            output = model(b_x)  # cnn final output
            preds = None
            _, preds = torch.max(output, 1) 
            
            criterion = af.get_loss_criterion('')
            stage_loss = criterion(output, b_y)  
            optimizer.zero_grad()           # clear gradients for this training step
            stage_loss.backward()           # backpropagation, compute gradients
            optimizer.step()   


            loss.append(stage_loss)
            label_list.append(b_y.detach().cpu().numpy())
            y_pred_list.append(preds.detach().cpu().numpy())
            sensitive_group_list.append(sensitive_group.numpy())

        scheduler.step()
        label_list = np.concatenate(label_list)
        y_pred_list = np.concatenate(y_pred_list)
        sensitive_group_list = np.concatenate(sensitive_group_list)    
        end_time = time.time()


        train_results = accuracy_metrices(label_list, y_pred_list, sensitive_group_list, sensitive_group_name_list)
        
        print('Training details')
        for k, v in train_results.items():
            print('{}:{:.4f}'.format(k, v))

        print('Loss: {}'.format(sum(loss) / len(loss)))
        epoch_time = int(end_time-start_time)
        print('Epoch took {} seconds.'.format(epoch_time))

        if epoch % 5 == 0:
            torch.save(model, '{}/{}.pth'.format(models_path, epoch))
            print('Start testing...')
            cnn_test_fairness(model, data.test_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_cnn')
    parser.add_argument('--fairness_attribute', type=str, default='',
                    help='fairness_attribute')
    parser.add_argument('--target_attribute', type=str, default='',
                    help='')
    parser.add_argument('--training_title', type=str, default='',
                    help='')
    parser.add_argument('--epochs', type=int, default=200,
                    help='')
    parser.add_argument('--lr', type=float, default=0.01,
                    help='')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='')
    args = parser.parse_args()
    training_title = args.training_title
    print(training_title)
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    models_path = 'networks/{}/{}'.format(af.get_random_seed(), training_title)
    tensor_board_path = 'runs/{}/train_models{}'.format(training_title, af.get_random_seed())
    af.create_path(models_path)
    af.create_path(tensor_board_path)
    af.create_path('outputs/{}'.format(training_title))
    af.set_logger('outputs/{}/train_models{}'.format(training_title, af.get_random_seed()))
    model = resnet18(class_num=2)
    model.to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    dataset = dataset_handler('celebA')(batch_size=args.batch_size, fairness_attribute=args.fairness_attribute, target_attribute=args.target_attribute)

    cnn_train(model, dataset, args.epochs, optimizer, scheduler, device, tensor_board_path, models_path, sensitive_group_name_list=['Female', 'Male'])