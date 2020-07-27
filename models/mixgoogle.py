from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import quant_module as qm

__all__ = [
    'mixgoogle_w1234a234',
]

_GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicQuantConv2d(nn.Module):

    def __init__(self, conv_func, in_channels, out_channels, **kwargs):
        super(BasicQuantConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)
        self.conv = conv_func(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class Inception(nn.Module):

    def __init__(self, conv_func, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red,
                 ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

        self.branch1 = conv_func(in_channels, ch1x1, bias=False, kernel_size=1, **kwargs)

        self.branch2_1 = conv_func(in_channels, ch3x3red, bias=False, kernel_size=1, **kwargs)
        self.branch2_2 = BasicQuantConv2d(
            conv_func, ch3x3red, ch3x3, kernel_size=3, padding=1, **kwargs)

        self.branch3_1 = conv_func(in_channels, ch5x5red, bias=False, kernel_size=1, **kwargs)
        self.branch3_2 = BasicQuantConv2d(
            conv_func, ch5x5red, ch5x5, kernel_size=5, padding=2, **kwargs)

        self.branch4 = BasicQuantConv2d(
            conv_func, in_channels, pool_proj, kernel_size=1, **kwargs)

    def forward(self, x):
        branch4 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1, ceil_mode=True)
        branch4 = self.branch4(branch4)

        x = self.bn(x)

        branch1 = self.branch1(x)

        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)

        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 2048
        x = F.dropout(x, 0.7, training=self.training)
        # N x 2048
        x = self.fc2(x)
        # N x 1024

        return x


class GoogLeNet(nn.Module):

    def __init__(self, conv_func, num_classes=1000, aux_logits=False,
                 init_weights=True, **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        super(GoogLeNet, self).__init__()
        self.conv_func = conv_func
        self.aux_logits = aux_logits
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicQuantConv2d(conv_func, 64, 64, kernel_size=1, **kwargs)
        self.conv3 = BasicQuantConv2d(conv_func, 64, 192, kernel_size=3, padding=1, **kwargs)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(conv_func, 192, 64, 96, 128, 16, 32, 32, **kwargs)
        self.inception3b = Inception(conv_func, 256, 128, 128, 192, 32, 96, 64, **kwargs)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(conv_func, 480, 192, 96, 208, 16, 48, 64, **kwargs)
        self.inception4b = Inception(conv_func, 512, 160, 112, 224, 24, 64, 64, **kwargs)
        self.inception4c = Inception(conv_func, 512, 128, 128, 256, 24, 64, 64, **kwargs)
        self.inception4d = Inception(conv_func, 512, 112, 144, 288, 32, 64, 64, **kwargs)
        self.inception4e = Inception(conv_func, 528, 256, 160, 320, 32, 128, 128, **kwargs)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(conv_func, 832, 256, 160, 320, 32, 128, 128, **kwargs)
        self.inception5b = Inception(conv_func, 832, 384, 192, 384, 48, 128, 128, **kwargs)
        self.inception5b_bn = nn.BatchNorm2d(1024, eps=0.001)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        x = self.inception5b_bn(x)
        x = F.relu(x, inplace=True)
        # N x 1024 x 7 x 7
        assert x.shape[2] == 7
        assert x.shape[3] == 7
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        return x

    def complexity_loss(self):
        size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                loss += m.complexity_loss()
                size_product += [m.size_product]
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            if isinstance(m, self.conv_func):
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = m.fetch_best_arch(layer_idx)
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_mixbitops += mixbitops
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw


def mixgoogle_w1234a234(**kwargs):
    return GoogLeNet(qm.MixActivConv2d, wbits=[1, 2, 3, 4], abits=[2, 3, 4], share_weight=True, **kwargs)
