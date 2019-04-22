#!/usr/bin/env python
# coding: utf-8

# ## Dropout is best !!!!111!1!1!

# In[1]:


import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from fashion import FashionMNIST
import torch.utils.model_zoo as model_zoo
from collections import namedtuple


# In[2]:


__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
           'resnet18', 'resnet34',]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    # vgg with TensorFlow
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    # ResNet with TensorFlow
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# In[3]:


train_data = FashionMNIST('../data', train=True, download=True,
                   transform=torchvision.transforms.Compose([
                       #torchvision.transforms.Grayscale(3),
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                   ]))

valid_data = FashionMNIST('../data', train=True, download=True,
                   transform=torchvision.transforms.Compose([
                       #torchvision.transforms.Grayscale(3),
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_idx = np.random.choice(train_data.data.shape[0], 54000, replace=False)

train_data.data = train_data.data[train_idx, :]
train_data.targets = train_data.targets[torch.from_numpy(train_idx).type(torch.LongTensor)]

mask = np.ones(60000)
mask[train_idx] = 0

valid_data.data = valid_data.data[torch.from_numpy(np.argwhere(mask)), :].squeeze()
valid_data.targets = valid_data.targets[torch.from_numpy(mask).type(torch.ByteTensor)]

batch_size = 100
test_batch_size = 100

train_loader = torch.utils.data.DataLoader(train_data,
    batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_data,
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    FashionMNIST('../data', train=False, transform=torchvision.transforms.Compose([
                       #torchvision.transforms.Grayscale(3),
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)

validation_data = {}
loss_train_data = {}
loss_validation_data = {}

#plt.imshow(train_loader.dataset.train_data[1].numpy()) #Potentially printing out an image


# In[ ]:


#plt.imshow(train_loader.dataset.train_data[10].numpy()) # another image


# #### Modèle 1: FCC sans dropout

# In[ ]:


class FccWithoutDropout(nn.Module):
    def __init__(self, model_id, input_size, hidden_sizes, output_size, activation_function):
        super().__init__()
        self.model_id = model_id
        self.activation_function = activation_function
        self.inputToHidden = nn.Linear(input_size, hidden_sizes[0]) #Linear(D, M)
        self.hiddenToHidden = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hiddenToHidden.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.hiddenToOutput = nn.Linear(hidden_sizes[len(hidden_sizes) - 1], output_size) #Linear(M, K)
    
    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = self.activation_function(self.inputToHidden(x))
        for f in self.hiddenToHidden:
            x = self.activation_function(f(x))
        x = F.log_softmax(self.hiddenToOutput(x), dim=1)
        return x


# #### Modèle 2: FCC avec dropout

# In[ ]:


class FccWithDropout(nn.Module):
    def __init__(self, model_id, input_size, hidden_sizes, output_size, activation_function, dropout):
        super().__init__()
        self.model_id = model_id
        self.activation_function = activation_function
        self.inputToHidden = nn.Linear(input_size, hidden_sizes[0]) #Linear(D, M)
        self.hiddenToHidden = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hiddenToHidden.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.hiddenToOutput = nn.Linear(hidden_sizes[len(hidden_sizes) - 1], output_size) #Linear(M, K)
        self.dropout = dropout
    
    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = self.activation_function(self.dropout(self.inputToHidden(x)))
        for f in self.hiddenToHidden:
            x = self.activation_function(self.dropout(f(x)))
        x = F.log_softmax(self.dropout(self.hiddenToOutput(x)), dim=1)
        return x


# #### Modèle 3: CNN sans dropout

# In[ ]:


class SimpleCnnWithoutDropout(nn.Module): #taken from https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
    #Our batch shape for input x is (1, 28, 28)
    
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        
        #Input channels = 1, output channels = 18
        self.conv1 = torch.nn.Conv2d(1, 25, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(25 * 14 * 14, 256)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(256, 10)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (1, 28, 28) to (25, 28, 28)
        x = F.relu(self.conv1(x))
        
        #Size changes from (25, 28, 28) to (25, 14, 14)
        x = self.pool(x)
       
        #Reshape data to input to the input layer of the neural net
        #Size changes from (25, 14, 14) to (1, 4900)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 25 * 14 *14)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4900) to (1, 256)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 256) to (1, 10)
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)
        return(x)


# #### Modèle 4: CNN avec dropout

# In[ ]:


class SimpleCnnWithDropout(nn.Module): #taken from https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
    #Our batch shape for input x is (1, 28, 28)
    
    def __init__(self, model_id, dropout):
        super().__init__()
        self.model_id = model_id
        self.dropout = dropout
        
        #Input channels = 1, output channels = 18
        self.conv1 = torch.nn.Conv2d(1, 25, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(25 * 14 * 14, 256)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(256, 10)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (1, 28, 28) to (25, 28, 28)
        x = F.relu(self.dropout(self.conv1(x)))
        
        #Size changes from (25, 28, 28) to (25, 14, 14)
        x = self.pool(x)
       
        #Reshape data to input to the input layer of the neural net
        #Size changes from (25, 14, 14) to (1, 4900)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 25 * 14 *14)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4900) to (1, 256)
        x = F.relu(self.dropout(self.fc1(x)))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 256) to (1, 10)
        x = self.fc2(x)
        x = F.log_softmax(self.dropout(x), dim = 1)
        return(x)


# #### Modèles 5 et 6: VGG avec et sans dropout

# In[ ]:


class VGG(nn.Module):

    def __init__(self, features, model_id, dropout=True, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.model_id = model_id
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if dropout:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else :
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Linear(4096, num_classes),
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim = 1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

# #### Modèles 9 et 10: Resnet avec et sans dropout

# In[ ]:


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockWithDropout(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlockWithDropout, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.dropout(out, training=self.training);
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.dropout(out, training=self.training);
        out = self.relu(out)

        return out

    
class BasicBlockWithoutDropout(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlockWithoutDropout, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetWithDropout(nn.Module):

    def __init__(self, block, layers, model_id, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNetWithDropout, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.model_id = model_id

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.dropout(x, training=self.training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim = 1)

        return x

class ResNetWithoutDropout(nn.Module):

    def __init__(self, block, layers, model_id, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNetWithoutDropout, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.model_id = model_id

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim = 1)

        return x

def resnet18(pretrained=False, dropout=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        dropout (bool): If True, returns a model using dropout
    """
    if dropout:
        model = ResNetWithDropout(BasicBlockWithDropout, [2, 2, 2, 2], **kwargs)
    else :
        model = ResNetWithoutDropout(BasicBlockWithoutDropout, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, dropout=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        dropout (bool): If True, returns a model using dropout
    """
    if dropout:
        model = ResNetWithDropout(BasicBlockWithDropout, [3, 4, 6, 3], **kwargs)
    else :
        model = ResNetWithoutDropout(BasicBlockWithoutDropout, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

# #### Entraînement

# In[ ]:


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        #data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += F.nll_loss(output, target, size_average=False).item()

    total_loss /= len(train_loader.dataset)
    loss_train_data[model.model_id].append(total_loss)

    return model


# #### Validation

# In[ ]:


def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        #data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        valid_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_loader.dataset)
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    loss_validation_data[model.model_id].append(valid_loss)
    return 1.0 * correct.item() / len(valid_loader.dataset) 


# #### Test

# In[ ]:


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        #data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# #### Expérimentation

# In[ ]:


def experiment(model, epochs=10, lr=0.001):
    best_precision = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    validation_data[model.model_id] = []
    loss_validation_data[model.model_id]= []
    loss_train_data[model.model_id] = []
    for epoch in range(1, epochs + 1):
        model = train(model, train_loader, optimizer)
        precision = valid(model, valid_loader)
        validation_data[model.model_id].append(precision)
        if precision > best_precision:
            best_precision = precision
            best_model = model
    return best_model, best_precision


# #### Définition des modèles utilisés

# In[ ]:


halfDropout = nn.Dropout(0.5)
fifthDropout = nn.Dropout(0.2)

fcc = FccWithoutDropout("fcc",28*28, [512], 10, F.sigmoid)
fccHalfDropout = FccWithDropout("fcc_d.5",28*28, [512], 10, F.sigmoid, halfDropout)
fccFifthDropout = FccWithDropout("fcc_d.2",28*28, [512], 10, F.sigmoid, fifthDropout)
cnn = SimpleCnnWithoutDropout("cnn")
cnnHalfDropout = SimpleCnnWithDropout("cnn_d.5", halfDropout)
cnnFifthDropout = SimpleCnnWithDropout("cnn_d.2", fifthDropout)
vgg16WithoutDropout = vgg16(dropout=False, model_id = "vgg", num_classes=10)
vgg16WithDropout = vgg16(dropout=True, model_id = "vgg_d", num_classes=10)
vgg16BN = vgg16_bn(dropout=False, model_id = "vgg_bn")
resNet18WithoutDropout = resnet18(dropout=False, model_id = "resnet18", num_classes=10)
resNet18WithDropout = resnet18(dropout=True, model_id = "resnet18_d", num_classes=10)
resNet34WithoutDropout = resnet34(dropout=False, model_id = "resnet34", num_classes=10)
resNet34WithDropout = resnet34(dropout=True, model_id = "resnet34_d", num_classes=10)

models = [
          fcc, 
          fccHalfDropout, 
          fccFifthDropout,
          cnn,
          cnnHalfDropout,
          cnnFifthDropout, 
          #vgg16WithoutDropout,
          #vgg16WithDropout, 
          #vgg16BN,
          resNet18WithoutDropout,
          resNet18WithDropout,
          resNet34WithoutDropout,
          resNet34WithDropout
         ]


# #### main

# In[ ]:


best_precision = 0
for model in models:
    model.cuda()  # if you have access to a gpu
    model, precision = experiment(model, epochs=20)
    print("final precision for model ", model.model_id, " is ", precision)
    if precision > best_precision:
        best_precision = precision
        best_model = model

test(best_model, test_loader)
print("The best model is: " + best_model.model_id)
for i, data in validation_data.items():
    if "fcc" in i:
        plt.plot(data, label = i)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy of FCC with and without dropout")
plt.show()

for i, data in validation_data.items():
    if "cnn" in i:
        plt.plot(data, label = i)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy of CNN with and without dropout")
plt.show()

for i, data in validation_data.items():
    if "vgg" in i:
        plt.plot(data, label = i)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy of VGG with and without dropout")
plt.show()

for i, data in validation_data.items():
    if "resnet" in i:
        plt.plot(data, label = i)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy of Resnet with and without dropout")
plt.show()

for i, data in loss_validation_data.items():
    if "fcc" in i:
        plt.plot(data, label = i)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (validation)")
plt.title("Validation losses of FCC with and without dropout")
plt.show()

for i, data in loss_validation_data.items():
    if "cnn" in i:
        plt.plot(data, label = i)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (validation)")
plt.title("Validation losses of CNN with and without dropout")
plt.show()

for i, data in loss_validation_data.items():
    if "vgg" in i:
        plt.plot(data, label = i)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (validation)")
plt.title("Validation losses of VGG with and without dropout")
plt.show()

for i, data in loss_validation_data.items():
    if "resnet" in i:
        plt.plot(data, label = i)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (validation)")
plt.title("Validation losses of Resnet with and without dropout")
plt.show()

for i, data in loss_train_data.items():
    if "fcc" in i:
        plt.plot(data, label = i+" training", linestyle = '--')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (training)")
plt.title("Training losses of FCC with and without dropout")
plt.show()

for i, data in loss_train_data.items():
    if "cnn" in i:
        plt.plot(data, label = i+" training", linestyle = '--')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (training)")
plt.title("Training losses of CNN with and without dropout")
plt.show()

for i, data in loss_train_data.items():
    if "vgg" in i:
        plt.plot(data, label = i+" training", linestyle = '--')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (training)")
plt.title("Training losses of VGG with and without dropout")
plt.show()

for i, data in loss_train_data.items():
    if "resnet" in i:
        plt.plot(data, label = i+" training", linestyle = '--')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (training)")
plt.title("Training losses of Resnet with and without dropout")
plt.show()

for i, data in validation_data.items():
    print (i + "'s final accuracy after training was %.4f;" % data[len(data) - 1])
for i, data in loss_validation_data.items():
    print (i + "'s final loss during validation was %.4f;" % data[len(data) - 1])
for i, data in loss_train_data.items():
    print (i + "'s final loss during training was %.4f;" % data[len(data) - 1])
# In[ ]:





# In[ ]:




