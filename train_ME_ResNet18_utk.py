import torch
import time
import data
import numpy as np
import argparse
import torch.nn as nn
import aux_funcs as af
import torch.optim as optim


from data import *
from tqdm import tqdm
from model import *
from torch.optim import SGD
from torchvision import models
from collections import Counter
from random import choice, shuffle
from eval_ME_fairprune import *
            
def cnn_training_step(model, optimizer, data, labels, device):
    coeff_list = [0.3, 0.6, 0.75, 0.9] # 0.1, 0.2, 0.3, 0.4, 1.0
    b_x = data.to(device)          # batch x
    b_y = labels.to(device)        # batch y
    output = model(b_x)            # cnn final output
    criterion = af.get_loss_criterion()
    loss = 0
    for idx in range(4):
        loss += coeff_list[idx]*criterion(output[idx], b_y)
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients
    return loss


def cnn_train(model, data, epochs, optimizer, scheduler, device='cuda', models_path=None, one_batch_dataset=None, dataset_name=''):
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}
    for epoch in range(1, epochs):
        loss = []
        cur_lr = af.get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.aug_train_loader
        else:
            train_loader = data.train_loader

        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        for x, y, gender in tqdm(train_loader):
            loss.append(cnn_training_step(model, optimizer, x, y, device))

        scheduler.step()
     
        print("Loss: {}".format(sum(loss) / len(loss)))
        if epoch % 5 == 0:
            torch.save(model, '{}/{}.pth'.format(models_path, epoch))
        if epoch % 5 == 0:
            if one_batch_dataset: #eval fairness score
                eval_sdn_celebA(model, one_batch_dataset, 'resnet', dataset_name='')
            
    return metrics

class ResNet18_Early_Exits_CelebA(nn.Module):
    """A resnet18 sdn tailored for celebA
    """
    def __init__(self, pretrained=True, class_num=2):
        super(ResNet18_Early_Exits_CelebA, self).__init__()
        self.f = nn.ModuleList()
        self.num_output = 4
        self.confidence_threshold = 0.8
        num_channel = {
                       'layer1':64, 
                       'layer2':128,
                       'layer3':256,
                       'layer4':512
                      }

        for name, module in models.resnet18(pretrained=pretrained).named_children():
            if isinstance(module, nn.Linear):
                continue
            if isinstance(module, nn.Sequential):
                exit_branch = nn.Sequential(
                                              nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(num_channel[name], class_num, bias=True)
                                           )
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
        
    def get_out_channels(self, module):
        for name, out_module in module.named_modules():
            if name == '1.bn2':
                return out_module.num_features

    def forward(self, x):
        early_exits_outputs = []
        for layer, early_exits_layer in self.f:
            x = layer(x)
            if early_exits_layer != None:
                early_exits_outputs.append(early_exits_layer(x))
        return early_exits_outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_sdn')
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
    parser.add_argument('--batch_size', type=int, default=512,
                    help='')
    parser.add_argument('--dataset', type=str, default='celebA',
                    help='')
    parser.add_argument('--model', type=str, default='resnet18',
                    help='')

    args = parser.parse_args()
    training_title = args.training_title
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    models_path = 'networks/{}/{}'.format(af.get_random_seed(), training_title)
    af.create_path(models_path)
    af.create_path('outputs/{}'.format(training_title))
    af.set_logger('outputs/{}/train_models{}'.format(training_title, af.get_random_seed()))
    
    dataset = None
    one_batch_dataset = None
    dataset = dataset_handler('celebA')(batch_size=args.batch_size, fairness_attribute=args.fairness_attribute, target_attribute=args.target_attribute)
    
    one_batch_dataset = dataset_handler('celebA')(batch_size=1, fairness_attribute=args.fairness_attribute, target_attribute=args.target_attribute)
    
    
    model = ResNet18_Early_Exits_CelebA(class_num=3)
    model.to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    cnn_train(model, dataset, args.epochs, optimizer, scheduler, device, models_path, one_batch_dataset, args.dataset)
    
    
