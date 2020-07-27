from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import quant_module as qm


__all__ = [
    'quantgoogle_2w2a', 'quantgoogle_cfg', 'quantgoogle_pretrained_cfg',
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

    def __init__(self, conv_func, in_channels, out_channels, wbit, abit, **kwargs):
        super(BasicQuantConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)
        self.conv = conv_func(in_channels, out_channels, wbit, abit, bias=False, **kwargs)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class Inception(nn.Module):

    def __init__(self, conv_func, archws, archas, in_channels, ch1x1, ch3x3red, ch3x3,
                 ch5x5red, ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__()
        assert len(archas) == 6
        assert len(archws) == 6
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

        self.branch1 = conv_func(
            in_channels, ch1x1, archws[0], archas[0], bias=False, kernel_size=1, **kwargs)

        self.branch2_1 = conv_func(
            in_channels, ch3x3red, archws[1], archas[1], bias=False, kernel_size=1, **kwargs)
        self.branch2_2 = BasicQuantConv2d(
            conv_func, ch3x3red, ch3x3, archws[2], archas[2], kernel_size=3, padding=1, **kwargs)

        self.branch3_1 = conv_func(
            in_channels, ch5x5red, archws[3], archas[3], bias=False, kernel_size=1, **kwargs)
        self.branch3_2 = BasicQuantConv2d(
            conv_func, ch5x5red, ch5x5, archws[4], archas[4], kernel_size=5, padding=2, **kwargs)

        self.branch4 = BasicQuantConv2d(
            conv_func, in_channels, pool_proj, archws[5], archas[5], kernel_size=1, **kwargs)

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


class QuantGoogLeNet(nn.Module):

    def __init__(self, conv_func, archws, archas, num_classes=1000, aux_logits=False,
                 init_weights=True, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))
        super(QuantGoogLeNet, self).__init__()
        self.conv_func = conv_func
        self.aux_logits = aux_logits
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicQuantConv2d(
            conv_func, 64, 64, archws[0], archas[0], kernel_size=1, **kwargs)
        self.conv3 = BasicQuantConv2d(
            conv_func, 64, 192, archws[1], archas[1], kernel_size=3, padding=1, **kwargs)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(
            conv_func, archws[2:8], archas[2:8], 192, 64, 96, 128, 16, 32, 32, **kwargs)
        self.inception3b = Inception(
            conv_func, archws[8:14], archas[8:14], 256, 128, 128, 192, 32, 96, 64, **kwargs)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(
            conv_func, archws[14:20], archas[14:20], 480, 192, 96, 208, 16, 48, 64, **kwargs)
        self.inception4b = Inception(
            conv_func, archws[20:26], archas[20:26], 512, 160, 112, 224, 24, 64, 64, **kwargs)
        self.inception4c = Inception(
            conv_func, archws[26:32], archas[26:32], 512, 128, 128, 256, 24, 64, 64, **kwargs)
        self.inception4d = Inception(
            conv_func, archws[32:38], archas[32:38], 512, 112, 144, 288, 32, 64, 64, **kwargs)
        self.inception4e = Inception(
            conv_func, archws[38:44], archas[38:44], 528, 256, 160, 320, 32, 128, 128, **kwargs)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(
            conv_func, archws[44:50], archas[44:50], 832, 256, 160, 320, 32, 128, 128, **kwargs)
        self.inception5b = Inception(
            conv_func, archws[50:], archas[50:], 832, 384, 192, 384, 48, 128, 128, **kwargs)
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

    def fetch_arch_info(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                bitops = size_product * m.abit * m.wbit
                bita = m.memory_size.item() * m.abit
                bitw = m.param_size * m.wbit
                weight_shape = list(m.conv.weight.shape)
                print('idx {} with shape {}, bitops: {:.3f}M * {} * {}, memory: {:.3f}K * {}, '
                      'param: {:.3f}M * {}'.format(layer_idx, weight_shape, size_product, m.abit, m.wbit,
                                                   memory_size, m.abit, m.param_size, m.wbit))
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                layer_idx += 1
        return sum_bitops, sum_bita, sum_bitw


def _load_arch(arch_path, names_nbits):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    best_arch, worst_arch = {}, {}
    for name in names_nbits.keys():
        best_arch[name], worst_arch[name] = [], []
    for name, params in state_dict.items():
        name = name.split('.')[-1]
        if name in names_nbits.keys():
            alpha = params.cpu().numpy()
            assert names_nbits[name] == alpha.shape[0]
            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())

    return best_arch, worst_arch


def quantgoogle_2w2a(arch_cfg_path, **kwargs):
    archas, archws = [2] * 56, [2] * 56
    assert len(archas) == 56
    assert len(archws) == 56
    return QuantGoogLeNet(qm.QuantActivConv2d, archws, archas, **kwargs)


def quantgoogle_pretrained_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    best_activ = [2, 0, 2, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1, 1, 0,
                  1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1,
                  1, 1, 1, 2, 1, 1, 0, 1, 0, 2]
    best_weight = [3, 0, 3, 2, 0, 3, 2, 3, 2, 0, 0, 2, 0, 0, 2, 3, 0, 3, 2, 3, 2, 2, 0,
                   3, 1, 2, 2, 0, 0, 3, 1, 2, 2, 0, 0, 3, 0, 2, 2, 0, 0, 3, 0, 2, 3, 3,
                   1, 3, 2, 3, 3, 3, 2, 3, 2, 3]
    archas = [abits[a] for a in best_activ]
    archws = [wbits[w] for w in best_weight]
    assert len(archas) == 56
    assert len(archws) == 56
    return QuantGoogLeNet(qm.QuantActivConv2d, archws, archas, **kwargs)


def quantgoogle_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 56
    assert len(archws) == 56
    return QuantGoogLeNet(qm.QuantActivConv2d, archws, archas, **kwargs)

