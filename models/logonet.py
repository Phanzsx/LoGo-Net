import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import os
import torchvision.models as models

__all__ = ['logonet18', 'logonet34', 'logonet50']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class de_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(de_conv, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # conv3x3(out_ch, out_ch),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//scale, in_ch//scale, kernel_size=3, stride=1, padding=1)

        self.conv = de_conv(in_ch, out_ch)
        self.dropout = nn.Dropout()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, math.ceil(diffY / 2),
                   diffX // 2, math.ceil(diffX / 2)), "constant", 0)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
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


class TAD_block(nn.Module):

    def __init__(self, in_dim):
        super(TAD_block, self).__init__()
        self.chanel_in = in_dim

        self.gate_conv = nn.Sequential(
            conv1x1(in_dim, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.query_conv = conv1x1(in_dim, in_dim // 8)
        self.key_conv = conv1x1(in_dim, in_dim // 8)
        self.value_conv = conv1x1(in_dim, in_dim)
        self.mask_conv = conv1x1(in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * H * W)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is H * W)
        """
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B * N * C
        proj_query -= proj_query.mean(1).unsqueeze(1)
        proj_key = self.key_conv(x).view(B, -1, H * W)  # B * C * N
        proj_key -= proj_key.mean(2).unsqueeze(2)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)  # B * N * N
        gate = self.gate_conv(x).view(B, -1, H * W)  # B * 1 * N
        attention = attention.permute(0, 2, 1) * gate
        proj_value = self.value_conv(x).view(B, -1, H * W)  # B * C * N
        proj_value = self.relu(proj_value)

        tissue = torch.bmm(proj_value, attention)
        tissue = tissue.view(B, C, H, W)

        proj_mask = self.mask_conv(x).view(B, -1, H * W)  # B * 1 * N
        mask = self.softmax(proj_mask)
        anatomy = torch.bmm(proj_value, mask.permute(0, 2, 1)).unsqueeze(-1)

        out = tissue + anatomy
        out = self.gamma * out + x
        return out, tissue


class logonet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2):
        super(logonet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.tad1 = TAD_block(64)
        self.tad2 = TAD_block(64)
        self.tad3 = TAD_block(128)
        self.tad4 = TAD_block(256)

        self.up4 = up(512 + 256, 256)
        self.up3 = up(256 + 128, 128)
        self.up2 = up(128 + 64, 64)
        self.up1 = up(64 + 64, 64, 1)

        self.outconv = conv3x3(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x, tissue1 = self.tad1(x1)

        x2 = self.layer1(x)
        x, tissue2 = self.tad2(x2)
        x3 = self.layer2(x)
        x, tissue3 = self.tad3(x3)
        x4 = self.layer3(x)
        x, tissue4 = self.tad4(x4)
        x5 = self.layer4(x)

        x = self.avgpool(x5)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        y = self.up4(x5, tissue4)
        y = self.up3(y, tissue3)
        y = self.up2(y, tissue2)
        y = self.up1(y, tissue1)
        y = self.outconv(y)

        return x, y


def load_pretrained(model, premodel):
    pretrained_dict = premodel.state_dict()
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

def logonet18(pretrained=False):
    model = logonet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        premodel = models.resnet18(pretrained=True)
        premodel.fc = nn.Linear(512, 2)
        model = load_pretrained(model, premodel)
    return model

def logonet34(pretrained=False):
    model = logonet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        premodel = models.resnet34(pretrained=True)
        premodel.fc = nn.Linear(512, 2)
        model = load_pretrained(model, premodel)
    return model

def logonet50(pretrained=False):
    model = logonet(BasicBlock, [4, 8, 8, 4])
    return model

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    images = torch.randn(1, 3, 224, 224)
    model = logonet18(pretrained=True)
    out_class, out_seg = model(images)
    print(out_class.shape, out_seg.shape)