from pickle import bytes_types
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import aux_funcs as af
import torch.optim as optim
import matplotlib.pyplot as plt

from data import *
from tqdm import tqdm
from model import *
from util.bce_acc import *
# from fairness_metric import *
from random import choice, shuffle

import torch
import numpy as np

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


def get_confidence(dataset_name='', logits=None):
    if dataset_name == 'celebA':
        sigmoid = nn.functional.sigmoid(logits)
        return sigmoid.cpu().numpy()
    else:
        softmax = nn.functional.softmax(logits, dim=0)
        return torch.max(softmax).cpu().numpy()

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
            start_time = time.time()
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

def cnn_test_fairness_utk(model, loader, device='cpu', model_name='', dataset_name='', eval_final=False):
    print(dataset_name)
    model.eval()
    top1 = AverageMeter()
    groupAcc=[]
    for i in range(3):
        saGroupAcc=[]
        for j in range(4):
            saGroupAcc.append(AverageMeter())
        groupAcc.append(saGroupAcc)
    with torch.no_grad():
        for batch in tqdm(loader):
            b_x = batch[0].cuda()
            b_y = batch[1].cuda()
            gender = batch[2].cuda()
            start_time = time.time()
            bsz = b_y.shape[0]
            output = model(b_x)
            pred = None
          
            _, pred = torch.max(output, 1)
            
            acc1, group_acc, group_num = val_accuracy(output, b_y, gender, 3, 4)
            top1.update(acc1, bsz)

            for i in range(3):
                for j in range(4):
                    groupAcc[i][j].update(group_acc[i][j],group_num[i][j])
           
    odds=0
    odds_num=0
    for i in range(3):
        for j in range(4):
            for k in range(j+1,4):
                odds_num+=1
                odds+=torch.abs(groupAcc[i][j].avg-groupAcc[i][k].avg)
    
    print('\n * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Equalized Odds {odds:.3f}'.format(odds=(odds/odds_num).item()))
    print(' * Group-wise accuracy')
    for i in range(3):
        string='    Target class '+str(i)+'\n    '
        for j in range(4):
            string+= '    Sensitive class '+str(j)+': {groupAcc.avg:.3f}'.format(groupAcc=groupAcc[i][j])
        print(string+'\n') 

def eval_cnn_celebA(model, one_batch_dataset, model_name, dataset_name=''):
    cnn_test_fairness(model, one_batch_dataset.test_loader, model_name=model_name, dataset_name=dataset_name)


def eval_cnn_utk(model, one_batch_dataset, model_name, dataset_name=''):
    cnn_test_fairness_utk(model, one_batch_dataset.test_loader, model_name=model_name, dataset_name=dataset_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_sdn')
    parser.add_argument('--fairness_attribute', type=str, default='',
                    help='fairness_attribute')
    parser.add_argument('--target_attribute', type=str, default='',
                    help='')
    parser.add_argument('--training_title', type=str, default='',
                    help='')
    parser.add_argument('--load_path', type=str, default='',
                    help='')
    parser.add_argument('--logger_path', type=str, default='',
                    help='')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='')
    parser.add_argument('--dataset', type=str, default='celebA',
                    help='')
    parser.add_argument('--model_name', type=str, default='resnet18',
                    help='')
    parser.add_argument('--eval_final', type=bool, default=True,
                    help='')
    parser.add_argument('--remove_image_list', type=bool, default=True,
                    help='')
    args = parser.parse_args()
    model = resnet18_fair_prune()
    model.load_state_dict(torch.load(args.load_path))
    model.cuda()
    remove_list = []
    one_batch_dataset = dataset_handler(args.dataset)(batch_size=256, fairness_attribute=args.fairness_attribute, target_attribute=args.target_attribute, remove_img=remove_list)

    if(args.dataset == 'celebA'):
        eval_cnn_celebA(model, one_batch_dataset, args.model_name, dataset_name='')
    else:
        eval_cnn_utk(model, one_batch_dataset, args.model_name, dataset_name='')

    