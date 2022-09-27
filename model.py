import torch
import numpy as np
import torch.nn as nn
from torchvision import models


def model_handler(model):
    if model == 'VGG_Early_Exits':
        return VGG_Early_Exits
    elif model == 'ResNet_Early_Exits':
        return ResNet_Early_Exits
    elif model == 'ResNet_Early_Exits':
        return ResNet_Early_Exits
    elif model == 'ResNet18_Early_Exits_CelebA':
        return ResNet18_Early_Exits_CelebA
    elif model == 'ResNet18_Early_Exits':
        return ResNet18_Early_Exits
    else:
        return None

def forward_features(model, x):
    """
    Dump each IC's feature map.
    Args:
        model (nn.module) : model
        x (torch.tensor) : model input
    """
    early_exits_outputs = []
    for layer, early_exits_layer in model.f:
        x = layer(x)
        if early_exits_layer != None:
            ic_out = early_exits_layer[0](x)
            ic_out = early_exits_layer[1](ic_out)
            ic_out = early_exits_layer[2](ic_out)
            ic_out = early_exits_layer[3](ic_out)
            ic_out = early_exits_layer[4](ic_out)
            early_exits_outputs.append(ic_out)
    final_out = x
    early_exits_outputs.append(final_out) # append final out
    return early_exits_outputs

class resnet18_mdf(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(resnet18_mdf, self).__init__()
        self.f = nn.ModuleList()
        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, class_num)
        for _, module in models.resnet18(pretrained=pretrained).named_children():
            if isinstance(module, nn.Linear):
                continue
            self.f.append(module)
        self.g = nn.Linear(512, class_num, bias=True)

    def forward(self, x):
        for layer in self.f:
            x = layer(x)
        final_out = self.g(torch.flatten(x, start_dim=1))

        return final_out, torch.flatten(x, start_dim=1)

class vgg11_bn(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(vgg11_bn, self).__init__()
        self.model = models.vgg11_bn()
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, class_num)

    def forward(self, x):
        return self.model(x)

class resnet18_fair_prune_ME(nn.Module):
    def __init__(self, encoder=None, ME=None, pretrained=True, class_num=2):
        super(resnet18_fair_prune_ME, self).__init__()
        self.f = nn.ModuleList()
        ic_list = []
        self.num_output = 5
        self.confidence_threshold = 0.8
        num_channel = {
                       'layer1':64, 
                       'layer2':128,
                       'layer3':256,
                       'layer4':512
                      }
        
        for layer, ic in ME.f:
            if ic:
                ic_list.append(ic)
        sq = 0
        
        for name, module in encoder.named_children():
            if isinstance(module, nn.Linear):
                self.g = module
            if isinstance(module, nn.Sequential):
                exit_branch = ic_list[sq]
                sq+=1
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
    def forward(self, x):
        early_exits_outputs = []
        for layer, early_exits_layer in self.f:
            x = layer(x)
            if early_exits_layer != None:
                early_exits_outputs.append(early_exits_layer(x))
        final_out = self.g(torch.flatten(x, start_dim=1))
        early_exits_outputs.append(final_out) # append final out
        return early_exits_outputs


class resnet18_fair_prune(nn.Module):
    def __init__(self, encoder=None, classifier=None, pretrained=True, class_num=2):
        super(resnet18_fair_prune, self).__init__()
        layer_list = []
        if encoder:
            for layer, ic_layer in encoder:
                layer_list.append(layer)
        else:
            for name, module in models.resnet18(pretrained=pretrained).named_children():
                if isinstance(module, nn.Linear):
                    continue
                else:
                    layer_list.append(module)
                    
        self.encoder = nn.Sequential(*layer_list)

        if classifier:
            self.classifier = classifier
        else:
            self.classifier = nn.Linear(512, class_num, bias=True)
        
    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(torch.flatten(x, start_dim=1))


class resnet18(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, class_num)

    def forward(self, x):
        return self.model(x)

class resnet34(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(resnet34, self).__init__()
        self.model = models.resnet34(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, class_num)

    def forward(self, x):
        return self.model(x)

class resnet50(nn.Module):
    def __init__(self, class_num=8, pretrained=True):
        super(resnet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, class_num)

    def forward(self, x):
        return self.model(x)


class VGG_Early_Exits(nn.Module):
    def __init__(self, pretrained=True, class_num=8):
        super(VGG_Early_Exits, self).__init__()
        self.f = nn.ModuleList()
        self.num_output = 5
        self.confidence_threshold = 0.8
        exit_branch_pos = [5, 10, 15, 20]
        num_channel = {
                       '5':128, 
                       '10':256,
                       '15':512,
                       '20':512
                      }

        for name, module in models.vgg11_bn(pretrained=pretrained).features.named_children():
            if int(name) in exit_branch_pos:
                exit_branch = nn.Sequential(nn.Conv2d(num_channel[name], 64, kernel_size=3, stride=1, padding=0, bias=True),
                                                    nn.AdaptiveAvgPool2d(output_size=(64)),
                                                    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, bias=True),
                                                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                                    nn.Flatten(),
                                                    nn.Linear(32, class_num, bias=True)
                                                   )
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
                
        self.f.append(nn.ModuleList([nn.Flatten(), None])) # Flatten the features
        for name, module in models.vgg11_bn(pretrained=True).avgpool.named_children():
            self.f.append(nn.ModuleList([module, None]))
            
        for name, module in models.vgg11_bn(pretrained=True).classifier.named_children():
            if name != '6': #if not the last layer
                self.f.append(nn.ModuleList([module, None]))
            else:           #for last layer
                final_layer = nn.Linear(4096, class_num, bias=True)
                self.f.append(nn.ModuleList([final_layer, None]))

    def forward(self, x):
        early_exits_outputs = []
        for layer, early_exits_layer in self.f:
            torch.cuda.empty_cache()
            x = layer(x)
            if early_exits_layer != None:
                early_exits_outputs.append(early_exits_layer(x))
        early_exits_outputs.append(x) # append final out
        return early_exits_outputs
    
    def early_exit(self, x):
        outputs = []
        confidences = []
        output_id = 0
        for layer, early_exits_layer in self.f:
            x = layer(x)
            if early_exits_layer != None:
                early_exit_out = early_exits_layer(x)
                outputs.append(early_exit_out)
                softmax = nn.functional.softmax(early_exit_out[0], dim=0)
                confidence = torch.max(softmax).cpu().numpy()
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return early_exit_out, output_id, is_early
                output_id += 1
        output = x
        outputs.append(output)
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax).cpu().numpy()
        confidences.append(confidence)
        max_confidence_output = np.argmax(confidences)
        is_early = False
        return outputs[max_confidence_output], max_confidence_output, is_early


class ResNet_Early_Exits(nn.Module):
    def __init__(self, pretrained=True, class_num=8):
        super(ResNet_Early_Exits, self).__init__()
        self.f = nn.ModuleList()
        self.num_output = 5
        self.confidence_threshold = 0.8

        for name, module in models.resnet18(pretrained=pretrained).named_children():
            if isinstance(module, nn.Linear):
                continue
    
            if isinstance(module, nn.Sequential):
                exit_branch = nn.Sequential(nn.Conv2d(self.get_out_channels(module), 256, kernel_size=7, stride=2, padding=3, bias=True),
                                              nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                              nn.Flatten(),
                                              nn.Linear(256, class_num, bias=True)
                                           )
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
        self.g = nn.Linear(512, class_num, bias=True)
        
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
        final_out = self.g(torch.flatten(x, start_dim=1))
        early_exits_outputs.append(final_out) # append final out
        return early_exits_outputs
    
    def early_exit(self, x):
        outputs = []
        confidences = []
        output_id = 0
        for layer, early_exits_layer in self.f:
            x = layer(x)
            if early_exits_layer != None:
                early_exit_out = early_exits_layer(x)
                outputs.append(early_exit_out)
                softmax = nn.functional.softmax(early_exit_out[0], dim=0)
                confidence = torch.max(softmax).cpu().numpy()
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return early_exit_out, output_id, is_early
                output_id += 1
                
        output = self.g(torch.flatten(x, start_dim=1))
        outputs.append(output)
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax).cpu().numpy()
        confidences.append(confidence)
        max_confidence_output = np.argmax(confidences)
        is_early = False
        return outputs[max_confidence_output], max_confidence_output, is_early

class ResNet18_Early_Exits_CelebA(nn.Module):
    """A resnet18 sdn tailored for celebA
    """
    def __init__(self, pretrained=True, class_num=8):
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
        # self.g = nn.Linear(512, class_num, bias=True) 
        
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
        # final_out = self.g(torch.flatten(x, start_dim=1))
        # early_exits_outputs.append(final_out) # append final out
        return early_exits_outputs
    
class ResNet18_Early_Exits(nn.Module):
    def __init__(self, pretrained=True, class_num=8):
        super(ResNet18_Early_Exits, self).__init__()
        self.f = nn.ModuleList()
        self.num_output = 5
        self.confidence_threshold = 0.8

        for name, module in models.resnet18(pretrained=pretrained).named_children():
            if isinstance(module, nn.Linear):
                continue
    
            if isinstance(module, nn.Sequential):
                exit_branch = nn.Sequential(nn.Conv2d(self.get_out_channels(module), 64, kernel_size=3, stride=1, padding=0, bias=True),
                                            nn.AdaptiveAvgPool2d(output_size=(64)),
                                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, bias=True),
                                            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                            nn.Flatten(),
                                            nn.Linear(32, class_num, bias=True)
                                           )
                self.f.append(nn.ModuleList([module, exit_branch]))
            else:
                self.f.append(nn.ModuleList([module, None]))
        self.g = nn.Linear(512, class_num, bias=True)
        
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
        final_out = self.g(torch.flatten(x, start_dim=1))
        early_exits_outputs.append(final_out) # append final out
        return early_exits_outputs
    
    def early_exit(self, x):
        outputs = []
        confidences = []
        output_id = 0
        for layer, early_exits_layer in self.f:
            x = layer(x)
            if early_exits_layer != None:
                early_exit_out = early_exits_layer(x)
                outputs.append(early_exit_out)
                softmax = nn.functional.softmax(early_exit_out[0], dim=0)
                confidence = torch.max(softmax).cpu().numpy()
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return early_exit_out, output_id, is_early
                output_id += 1
                
        output = self.g(torch.flatten(x, start_dim=1))
        outputs.append(output)
        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax).cpu().numpy()
        confidences.append(confidence)
        max_confidence_output = np.argmax(confidences)
        is_early = False
        return outputs[max_confidence_output], max_confidence_output, is_early
    

