import torch
import torch.nn as nn
import math
from . import quant_module as qm

__all__ = [
    'quantres18_2w2a', 'quantres18_cfg', 'quantres18_pretrained_cfg',
    'quantres50_2w2a', 'quantres50_cfg', 'quantres50_pretrained_cfg',
]


class BasicBlock(nn.Module):
    expansion = 1
    num_layers = 2

    def __init__(self, conv_func, inplanes, planes, archws, archas, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(BasicBlock, self).__init__()
        assert len(archws) == 2
        assert len(archas) == 2
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv_func(inplanes, planes, archws[0], archas[0], kernel_size=3, stride=stride,
                               padding=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv2 = conv_func(planes, planes, archws[1], archas[1], kernel_size=3, stride=1,
                               padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.bn0(x)
        if self.downsample is not None:
            residual = out
        else:
            residual = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4
    num_layers = 3

    def __init__(self, conv_func, inplanes, planes, archws, archas, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(Bottleneck, self).__init__()
        assert len(archws) == 3
        assert len(archas) == 3
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv_func(inplanes, planes, archws[0], archas[0], kernel_size=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv2 = conv_func(planes, planes, archws[1], archas[1], kernel_size=3, stride=stride,
                               padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv3 = conv_func(
            planes, planes * 4, archws[2], archas[2], kernel_size=1, bias=False, **kwargs)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = self.bn0(x)
        if self.downsample is not None:
            residual = out
        else:
            residual = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self, block, conv_func, layers, archws, archas, num_classes=1000,
                 bnaff=True, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))
        self.inplanes = 64
        self.conv_func = conv_func
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=bnaff)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        eid = block.num_layers * layers[0] + (block.expansion > 1)
        self.layer1 = self._make_layer(block, conv_func, 64, layers[0], archws[:eid], archas[:eid],
                                       bnaff=bnaff, **kwargs)
        sid = eid
        eid = sid + block.num_layers * layers[1] + 1
        self.layer2 = self._make_layer(
            block, conv_func, 128, layers[1], archws[sid:eid], archas[sid:eid],
            stride=2, bnaff=bnaff, **kwargs
        )
        sid = eid
        eid = sid + block.num_layers * layers[2] + 1
        self.layer3 = self._make_layer(
            block, conv_func, 256, layers[2], archws[sid:eid], archas[sid:eid],
            stride=2, bnaff=bnaff, **kwargs
        )
        sid = eid
        self.layer4 = self._make_layer(block, conv_func, 512, layers[3], archws[sid:], archas[sid:],
                                       stride=2, bnaff=bnaff, **kwargs)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, conv_func, planes, blocks, archws, archas, stride=1,
                    bnaff=True, **kwargs):
        downsample = None
        interval = block.num_layers
        if stride != 1 or self.inplanes != planes * block.expansion:
            # the last element in arch is for downsample layer
            downsample = nn.Sequential(
                conv_func(self.inplanes, planes * block.expansion, archws[interval], archas[interval],
                          kernel_size=1, stride=stride, bias=False, **kwargs),
                nn.BatchNorm2d(planes * block.expansion),
            )
            archws.pop(interval)
            archas.pop(interval)

        layers = []
        assert len(archws) == blocks * interval
        assert len(archas) == blocks * interval
        layers.append(block(conv_func, self.inplanes, planes, archws[:interval], archas[:interval], stride,
                            downsample, bnaff=bnaff, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            sid, eid = interval * i, interval * (i + 1)
            layers.append(block(conv_func, self.inplanes, planes, archws[sid:eid], archas[sid:eid],
                                bnaff=bnaff, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(x)
        assert x.shape[2] == 7
        assert x.shape[3] == 7
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

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
                      'param: {:.3f}M * {}'.format(layer_idx, weight_shape, size_product, m.abit,
                                                   m.wbit, memory_size, m.abit, m.param_size, m.wbit))
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


def quantres18_2w2a(arch_cfg_path, **kwargs):
    archas, archws = [2] * 19, [2] * 19
    assert len(archas) == 19
    assert len(archws) == 19
    return ResNet(BasicBlock, qm.QuantActivConv2d, [2, 2, 2, 2], archws, archas, **kwargs)


def quantres18_pretrained_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    best_activ = [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    best_weight = [0, 0, 0, 0, 2, 1, 3, 1, 0, 2, 1, 3, 1, 0, 2, 1, 3, 1, 1]
    archas = [abits[a] for a in best_activ]
    archws = [wbits[w] for w in best_weight]
    assert len(archas) == 19
    assert len(archws) == 19
    return ResNet(BasicBlock, qm.QuantActivConv2d, [2, 2, 2, 2], archws, archas, **kwargs)


def quantres18_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 19
    assert len(archws) == 19
    return ResNet(BasicBlock, qm.QuantActivConv2d, [2, 2, 2, 2], archws, archas, **kwargs)


def quantres50_2w2a(arch_cfg_path, **kwargs):
    archas, archws = [2] * 52, [2] * 52
    assert len(archas) == 52
    assert len(archws) == 52
    return ResNet(Bottleneck, qm.QuantActivConv2d, [3, 4, 6, 3], archws, archas, **kwargs)


def quantres50_pretrained_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    best_activ = [2, 1, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1,
                  0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1]
    best_weight = [3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   2, 1, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1,
                   0, 1, 2, 2, 1, 3]
    archas = [abits[a] for a in best_activ]
    archws = [wbits[w] for w in best_weight]
    assert len(archas) == 52
    assert len(archws) == 52
    return ResNet(Bottleneck, qm.QuantActivConv2d, [3, 4, 6, 3], archws, archas, **kwargs)


def quantres50_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 52
    assert len(archws) == 52
    return ResNet(Bottleneck, qm.QuantActivConv2d, [3, 4, 6, 3], archws, archas, **kwargs)
