import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
affine_par = True

#from modules.high_dim_filter_loader import HighDimFilterModule
import torch.nn.functional as F
from torch.autograd import Variable

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class Residual_Covolution(nn.Module):
    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg

class Residual_Refinement_Module(nn.Module):

    def __init__(self, num_classes):
        super(Residual_Refinement_Module, self).__init__()
        self.RC1 = Residual_Covolution(2048, 512, num_classes)
        self.RC2 = Residual_Covolution(2048, 512, num_classes)

    def forward(self, x):
        x, seg1 = self.RC1(x)
        _, seg2 = self.RC2(x)
        return [seg1, seg1+seg2]

class ResNet_Refine(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_Refine, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = Residual_Refinement_Module(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

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
        x = self.layer5(x)

        return x     

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        embedding = x
        x = self.layer5(x)

        return x, embedding

class MS_Deeplab(nn.Module):
    def __init__(self,block,num_classes):
        super(MS_Deeplab,self).__init__()
        self.Scale = ResNet(block,[3, 4, 23, 3],num_classes)   #changed to fix #4 

    def forward(self,x):
        output = self.Scale(x) # for original scale
        output_size = output.size()[2]
        input_size = x.size()[2]

        self.interp1 = nn.Upsample(size=(int(input_size*0.75)+1, int(input_size*0.75)+1), mode='bilinear')
        self.interp2 = nn.Upsample(size=(int(input_size*0.5)+1, int(input_size*0.5)+1), mode='bilinear')
        self.interp3 = nn.Upsample(size=(output_size, output_size), mode='bilinear')

        x75 = self.interp1(x)
        output75 = self.interp3(self.Scale(x75)) # for 0.75x scale

        x5 = self.interp2(x)
        output5 = self.interp3(self.Scale(x5))	# for 0.5x scale

        out_max = torch.max(torch.max(output, output75), output5)
        return [output, output75, output5, out_max]

def Res_Ms_Deeplab(num_classes=21):
    model = MS_Deeplab(Bottleneck, num_classes)
    return model

def Res_Deeplab(num_classes=21, is_refine=False):
    if is_refine:
        model = ResNet_Refine(Bottleneck,[3, 4, 23, 3], num_classes)
    else:
        model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model


import pdb


class ResNetSiam(nn.Module):
    def __init__(self, block, layers, emb_size):
        self.inplanes = 64
        self.emb_size = emb_size
        super(ResNetSiam, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(2048, 128, 3, stride=2, padding=1),
            #nn.BatchNorm2d(128, affine=affine_par),
            #nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            #nn.BatchNorm2d(128, affine=affine_par),
            nn.Upsample(size=self.emb_size, mode='bilinear'),
            #nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1),
            #nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        )

        self.bnout = nn.BatchNorm2d(1, affine=affine_par)

        self.merge_classifier = nn.Sequential(
            nn.Conv2d(1, 1, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x1, x2, selected_pixels, embedding1=None, flag=1, selected_pixels_2=None):
        if flag:
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.maxpool(x1)
            x1 = self.layer1(x1)
            x1 = self.layer2(x1)
            x1 = self.layer3(x1)
            x1 = self.layer4(x1)
            embedding1 = self.decoder(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        embedding2 = self.decoder(x2)

        n, d, h, w = embedding1.size()
        x1 = embedding1.view(n, d, -1)#permute(1, 2, 0).contiguous().view(-1, d)
        x2 = embedding2.view(n, d, -1)#permute(1, 2, 0).contiguous().view(-1, d)
        x1 = x1 / torch.norm(x1, dim=1).view(x1.size()[0], 1, x1.size()[-1])
        x2 = x2 / torch.norm(x2, dim=1).view(x2.size()[0], 1, x2.size()[-1])
        if selected_pixels_2 is None:
            y = torch.matmul(x1[:, :, selected_pixels].transpose(1,2), x2)
        else:
            y = torch.matmul(x1[:, :, selected_pixels].transpose(1, 2), x2[:, :, selected_pixels_2])
        #y = ((x1.unsqueeze(1) - x2.unsqueeze(0))**2 + 1e-10).sum(-1).sqrt()
        y = y.unsqueeze(1)
        y = self.bnout(y)
        y = self.merge_classifier(y)


        return y, embedding1


def ResSiam_Deeplab(emb_size=(240, 427)):
    model = ResNetSiam(Bottleneck,[3, 4, 23, 3], emb_size=emb_size)
    return model

class ResNetSiam_old_harkirat(nn.Module):
    def __init__(self, block, layers, emb_size):
        self.inplanes = 64
        self.emb_size = emb_size
        super(ResNetSiam_old_harkirat, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(2048, 128, 3, stride=2, padding=1),
            #nn.BatchNorm2d(128, affine=affine_par),
            #nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            #nn.BatchNorm2d(128, affine=affine_par),
            nn.Upsample(size=self.emb_size, mode='bilinear'),
            #nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1),
            #nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        )

        self.bnout = nn.BatchNorm2d(1, affine=affine_par)

        self.merge_classifier = nn.Sequential(
            nn.Conv2d(1, 1, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x1, x2, selected_pixels, embedding1=None, flag=1, selected_pixels_2=None):
        if flag:
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.maxpool(x1)
            x1 = self.layer1(x1)
            x1 = self.layer2(x1)
            x1 = self.layer3(x1)
            x1 = self.layer4(x1)
            embedding1 = self.decoder(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        embedding2 = self.decoder(x2)

        n, d, h, w = embedding1.size()
        x1 = embedding1.view(n, d, -1)#permute(1, 2, 0).contiguous().view(-1, d)
        x2 = embedding2.view(n, d, -1)#permute(1, 2, 0).contiguous().view(-1, d)
        if selected_pixels_2 is None:
            y = torch.matmul(x1[:, :, selected_pixels].transpose(1,2), x2)
        else:
            y = torch.matmul(x1[:, :, selected_pixels].transpose(1, 2), x2[:, :, selected_pixels_2])
        #y = ((x1.unsqueeze(1) - x2.unsqueeze(0))**2 + 1e-10).sum(-1).sqrt()
        y = y.unsqueeze(1)
        y = self.bnout(y)
        #y = self.merge_classifier(y)


        return y, embedding1


def ResSiam_Deeplab_old_harkirat(emb_size=(240, 427)):
    model = ResNetSiam_old_harkirat(Bottleneck,[3, 4, 23, 3], emb_size=emb_size)
    return model

class ResNetUpsampled(nn.Module):
    def __init__(self, block, layers, emb_size):
        self.inplanes = 64
        self.emb_size = emb_size
        super(ResNetUpsampled, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 128, 3, stride=2, padding=1),
            #nn.BatchNorm2d(128, affine=affine_par),
            #nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            #nn.BatchNorm2d(128, affine=affine_par),
            nn.Upsample(size=self.emb_size[1:], mode='bilinear'),
            #nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1),
            #nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        )

        self.bnout = nn.BatchNorm2d(1, affine=affine_par)

        self.merge_classifier = nn.Sequential(
            nn.Conv2d(1, 1, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        embedding = self.decoder(x)

        return embedding


def ResNetUpsampled_Deeplab(emb_size=(128, 240, 427)):
    model = ResNetUpsampled(Bottleneck,[3, 4, 23, 3], emb_size=emb_size)
    return model

'''
class HighDimFilterNetwork(nn.Module):
    def __init__(self, bilateral, theta_alpha, theta_beta, theta_gamma):
        super(HighDimFilterNetwork, self).__init__()

        self.highDimFilter = HighDimFilterModule(bilateral, theta_alpha, theta_beta, theta_gamma)

    def forward(self, input_tensor, image_tensor):
        return self.highDimFilter(input_tensor, image_tensor)

class ResNetSiam_End2End_TrainableMat(nn.Module):
    def __init__(self, block, layers, num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations, gpu, requires_flow):
        self.inplanes = 64
        super(ResNetSiam_End2End_TrainableMat, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 128, 3, stride=2, padding=1),
            nn.Upsample(size=(240, 427), mode='bilinear'),
        )

        self.bnout = nn.BatchNorm2d(1, affine=affine_par)

        self.merge_classifier = nn.Sequential(
            nn.Conv2d(1, 1, 1),
        )

        self.image_dims = None
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.requires_flow = requires_flow
        self.spatial_ker_weights = nn.Linear(self.num_classes, self.num_classes, bias=False)
        self.bilateral_ker_weights = nn.Linear(self.num_classes, self.num_classes, bias=False)
        self.compatibility_matrix = nn.Linear(self.num_classes, self.num_classes, bias=False)
        self.spatial_norm_vals_layer = HighDimFilterNetwork(bilateral=False, theta_alpha=1.0, theta_beta=1.0, theta_gamma=self.theta_gamma)
        self.spatial_out_layer = HighDimFilterNetwork(bilateral=False, theta_alpha=1.0, theta_beta=1.0, theta_gamma=self.theta_gamma)
        self.bilateral_norm_vals_layer = HighDimFilterNetwork(bilateral=True, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta, theta_gamma=1.0)
        self.bilateral_out_layer = HighDimFilterNetwork(bilateral=True, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta, theta_gamma=1.0)
        if self.requires_flow:
            self.flow_ker_weights = nn.Linear(self.num_classes, self.num_classes, bias=False)
            self.flow_norm_vals_layer = HighDimFilterNetwork(bilateral=True, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta, theta_gamma=1.0)
            self.flow_out_layer = HighDimFilterNetwork(bilateral=True, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta, theta_gamma=1.0)
        
        self.softmax_layer = nn.Softmax(dim=0)
        self.gpu = gpu

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.uniform_(-0.05, 0.05)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def _step(self, q_values, rgb, flow, spatial_norm_vals, bilateral_norm_vals, flow_norm_vals, unaries):
        softmax_out = self.softmax_layer(q_values)
        spatial_out = self.spatial_out_layer(softmax_out, rgb)
        spatial_out = spatial_out / spatial_norm_vals
        bilateral_out = self.bilateral_out_layer(softmax_out, rgb)
        bilateral_out = bilateral_out / bilateral_norm_vals

        message_passing = self.spatial_ker_weights(torch.transpose(spatial_out.cuda(self.gpu).view((self.num_classes, -1)), 0, 1)) + self.bilateral_ker_weights(torch.transpose(bilateral_out.cuda(self.gpu).view((self.num_classes, -1)), 0, 1))
        if self.requires_flow:
            flow_out = self.flow_out_layer(softmax_out, flow)
            flow_out = flow_out / flow_norm_vals
            message_passing += self.flow_ker_weights(torch.transpose(flow_out.cuda(self.gpu).view((self.num_classes, -1)), 0, 1))

        pairwise = torch.transpose(self.compatibility_matrix(message_passing), 0, 1)
        pairwise = pairwise.contiguous().view((self.num_classes, self.image_dims[0], self.image_dims[1]))
        q_values = unaries - pairwise.cpu()
        return q_values

    def forward(self, x1, x2, selected_pixels, class_pixels, image_dims, flow=None, embedding1=None, flag=1, num_iterations=None, selected_pixels_2=None):
        if num_iterations != None:
            self.num_iterations = num_iterations
        self.image_dims = image_dims
        rgb = x2.squeeze(0).cpu()
        if flag:
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.maxpool(x1)
            x1 = self.layer1(x1)
            x1 = self.layer2(x1)
            x1 = self.layer3(x1)
            x1 = self.layer4(x1)
            embedding1 = self.decoder(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        embedding2 = self.decoder(x2)

        n, d, h, w = embedding1.size()
        x1 = embedding1.view(n, d, -1)#permute(1, 2, 0).contiguous().view(-1, d)
        x2 = embedding2.view(n, d, -1)#permute(1, 2, 0).contiguous().view(-1, d)
        if selected_pixels_2 is None:
            y = torch.matmul(x1[:, :, selected_pixels].transpose(1,2), x2)
        else:
            y = torch.matmul(x1[:, :, selected_pixels].transpose(1, 2), x2[:, :, selected_pixels_2])
        #y = ((x1.unsqueeze(1) - x2.unsqueeze(0))**2 + 1e-10).sum(-1).sqrt()
        y = y.unsqueeze(1)
        y = self.bnout(y)
        y = self.merge_classifier(y)

        #crf_part
        mean_x1_tostack = []
        for class_pixels_i in class_pixels:
            mean_x1_tostack.append(x1[:, :, class_pixels_i][0].mean(1))

        mean_x1 = torch.stack(mean_x1_tostack)
        scores_picked_tostack = []
        for i in range(len(class_pixels)):
            scores_picked_tostack.append(torch.matmul(x2[0].transpose(1,0), mean_x1[i]))

        scores_picked = torch.stack(scores_picked_tostack)        
        output = scores_picked.view(1,-1,240,427)
        output = F.upsample(output, (self.image_dims[0], self.image_dims[1]), mode='bilinear') # H x W
        output = F.softmax(output, dim=1)

        unaries = output.squeeze(0).cpu()
        all_ones = Variable(torch.ones([self.num_classes, self.image_dims[0], self.image_dims[1]]))

        spatial_norm_vals = self.spatial_norm_vals_layer(all_ones, rgb)
        bilateral_norm_vals = self.bilateral_norm_vals_layer(all_ones, rgb)
        flow_norm_vals = None
        if self.requires_flow:
            flow_norm_vals = self.flow_norm_vals_layer(all_ones, rgb)
        
        q_values = unaries

        for i in range(self.num_iterations):
            q_values = self._step(q_values, rgb, flow, spatial_norm_vals, bilateral_norm_vals, flow_norm_vals, unaries) #q_values aren't normalized along channels. They may sum to nearly 1.03 along channel but most values can be negative.

        return y, embedding1, output, q_values.view((1, self.num_classes, self.image_dims[0], self.image_dims[1])).cuda()


def ResSiam_Deeplab_End2End_TrainableMat(num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations, gpu, requires_flow=False):
    model = ResNetSiam_End2End_TrainableMat(Bottleneck,[3, 4, 23, 3], num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations, gpu, requires_flow)
    return model

class Scalar(nn.Module):
    def __init__(self, gpu):
        super(Scalar, self).__init__()
        self.scalar = nn.Parameter(torch.FloatTensor(1).cuda(gpu), requires_grad=True)

    def forward(self, input_tensor):
        return self.scalar * input_tensor


class ResNetSiam_End2End(nn.Module):
    def __init__(self, block, layers, theta_alpha, theta_beta, theta_gamma, num_iterations, gpu, requires_flow):
        self.inplanes = 64
        super(ResNetSiam_End2End, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 128, 3, stride=2, padding=1),
            nn.Upsample(size=(240, 427), mode='bilinear'),
        )

        self.bnout = nn.BatchNorm2d(1, affine=affine_par)

        self.merge_classifier = nn.Sequential(
            nn.Conv2d(1, 1, 1),
        )

        self.image_dims = None
        self.num_classes = None
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.requires_flow = requires_flow
        # self.spatial_ker_weights = nn.Conv1d(1, 1, 1, bias=False)
        # self.bilateral_ker_weights = nn.Conv1d(1, 1, 1, bias=False)
        # self.flow_ker_weights = nn.Conv1d(1, 1, 1, bias=False)
        self.spatial_ker_weights = Scalar(gpu)
        self.bilateral_ker_weights = Scalar(gpu)

        self.compatibility_matrix = None
        self.spatial_norm_vals_layer = HighDimFilterNetwork(bilateral=False, theta_alpha=1.0, theta_beta=1.0, theta_gamma=self.theta_gamma)
        self.spatial_out_layer = HighDimFilterNetwork(bilateral=False, theta_alpha=1.0, theta_beta=1.0, theta_gamma=self.theta_gamma)
        self.bilateral_norm_vals_layer = HighDimFilterNetwork(bilateral=True, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta, theta_gamma=1.0)
        self.bilateral_out_layer = HighDimFilterNetwork(bilateral=True, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta, theta_gamma=1.0)
        if self.requires_flow:
            self.flow_ker_weights = Scalar(gpu)
            self.flow_norm_vals_layer = HighDimFilterNetwork(bilateral=True, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta, theta_gamma=1.0)
            self.flow_out_layer = HighDimFilterNetwork(bilateral=True, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta, theta_gamma=1.0)
        self.softmax_layer = nn.Softmax(dim=0)
        self.gpu = gpu

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, Scalar):
                m._parameters['scalar'].data.uniform_(0,0.05)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def _step(self, q_values, rgb, flow, spatial_norm_vals, bilateral_norm_vals, flow_norm_vals, unaries):
        softmax_out = self.softmax_layer(q_values)
        spatial_out = self.spatial_out_layer(softmax_out, rgb)
        spatial_out = spatial_out / spatial_norm_vals
        bilateral_out = self.bilateral_out_layer(softmax_out, rgb)
        bilateral_out = bilateral_out / bilateral_norm_vals

        #message_passing = torch.matmul(self.spatial_ker_weights, spatial_out.view((self.num_classes, -1))) + torch.matmul(self.bilateral_ker_weights, bilateral_out.view((self.num_classes, -1)))
        #pairwise = torch.matmul(self.compatibility_matrix, message_passing)
        message_passing = self.spatial_ker_weights(spatial_out.view((1,1,-1)).cuda(self.gpu)) + self.bilateral_ker_weights(bilateral_out.view((1,1,-1)).cuda(self.gpu))
        if self.requires_flow:
            flow_out = self.flow_out_layer(softmax_out, flow)
            flow_out = flow_out / flow_norm_vals
            message_passing += self.flow_ker_weights(flow_out.view((1,1,-1)).cuda(self.gpu))

        pairwise = -message_passing.cpu()
        pairwise = pairwise.view((self.num_classes, self.image_dims[0], self.image_dims[1]))
        q_values = unaries - pairwise
        return q_values

    def forward(self, x1, x2, selected_pixels, class_pixels, num_classes, image_dims, flow=None, embedding1=None, flag=1, num_iterations=None, selected_pixels_2=None):
        if num_iterations != None:
            self.num_iterations = num_iterations
        rgb = x2.squeeze(0).cpu()
        self.num_classes = num_classes
        self.image_dims = image_dims
        #self.compatibility_matrix = Variable(-torch.diag(torch.ones(self.num_classes)))
        if flag:
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.maxpool(x1)
            x1 = self.layer1(x1)
            x1 = self.layer2(x1)
            x1 = self.layer3(x1)
            x1 = self.layer4(x1)
            embedding1 = self.decoder(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        embedding2 = self.decoder(x2)

        n, d, h, w = embedding1.size()
        x1 = embedding1.view(n, d, -1)#permute(1, 2, 0).contiguous().view(-1, d)
        x2 = embedding2.view(n, d, -1)#permute(1, 2, 0).contiguous().view(-1, d)
        if selected_pixels_2 is None:
            y = torch.matmul(x1[:, :, selected_pixels].transpose(1,2), x2)
        else:
            y = torch.matmul(x1[:, :, selected_pixels].transpose(1, 2), x2[:, :, selected_pixels_2])
        #y = ((x1.unsqueeze(1) - x2.unsqueeze(0))**2 + 1e-10).sum(-1).sqrt()
        y = y.unsqueeze(1)
        y = self.bnout(y)
        y = self.merge_classifier(y)

        #crf_part

        mean_x1_tostack = []
        for class_pixels_i in class_pixels:
            mean_x1_tostack.append(x1[:, :, class_pixels_i][0].mean(1))

        mean_x1 = torch.stack(mean_x1_tostack)
        scores_picked_tostack = []
        for i in range(len(class_pixels)):
            scores_picked_tostack.append(torch.matmul(x2[0].transpose(1,0), mean_x1[i]))

        scores_picked = torch.stack(scores_picked_tostack)        
        output = scores_picked.view(1,-1,240,427)
        output = F.upsample(output, (self.image_dims[0], self.image_dims[1]), mode='bilinear') # H x W
        output = F.softmax(output, dim=1)

        unaries = output.squeeze(0).cpu()
        all_ones = Variable(torch.ones([self.num_classes, self.image_dims[0], self.image_dims[1]]))

        spatial_norm_vals = self.spatial_norm_vals_layer(all_ones, rgb)
        bilateral_norm_vals = self.bilateral_norm_vals_layer(all_ones, rgb)
        flow_norm_vals = None
        if self.requires_flow:
            flow_norm_vals = self.flow_norm_vals_layer(all_ones, rgb)
        
        q_values = unaries

        for i in range(self.num_iterations):
            q_values = self._step(q_values, rgb, flow, spatial_norm_vals, bilateral_norm_vals, flow_norm_vals, unaries) #q_values aren't normalized along channels. They may sum to nearly 1.03 along channel but most values can be negative.

        return y, embedding1, output, q_values.view((1, self.num_classes, self.image_dims[0], self.image_dims[1])).cuda()


def ResSiam_Deeplab_End2End(theta_alpha, theta_beta, theta_gamma, num_iterations, gpu, requires_flow=False):
    model = ResNetSiam_End2End(Bottleneck,[3, 4, 23, 3], theta_alpha, theta_beta, theta_gamma, num_iterations, gpu, requires_flow)
    return model

'''