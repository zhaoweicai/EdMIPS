import torch
import torch.nn as nn
import torch.nn.functional as F
from . import quant_module as qm


__all__ = [
    'quantinception_2w2a', 'quantinception_cfg', 'quantinception_pretrained_cfg',
]


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


class InceptionA(nn.Module):

    def __init__(self, conv_func, archws, archas, in_channels, pool_features, **kwargs):
        super(InceptionA, self).__init__()
        assert len(archas) == 7
        assert len(archws) == 7
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

        self.branch1x1 = conv_func(
            in_channels, 64, archws[0], archas[0], bias=False, kernel_size=1, **kwargs)

        self.branch5x5_1 = conv_func(
            in_channels, 48, archws[1], archas[1], bias=False, kernel_size=1, **kwargs)
        self.branch5x5_2 = BasicQuantConv2d(
            conv_func, 48, 64, archws[2], archas[2], kernel_size=5, padding=2, **kwargs)

        self.branch3x3dbl_1 = conv_func(
            in_channels, 64, archws[3], archas[3], bias=False, kernel_size=1, **kwargs)
        self.branch3x3dbl_2 = BasicQuantConv2d(
            conv_func, 64, 96, archws[4], archas[4], kernel_size=3, padding=1, **kwargs)
        self.branch3x3dbl_3 = BasicQuantConv2d(
            conv_func, 96, 96, archws[5], archas[5], kernel_size=3, padding=1, **kwargs)

        self.branch_pool = BasicQuantConv2d(
            conv_func, in_channels, pool_features, archws[6], archas[6], kernel_size=1, **kwargs)

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        x = self.bn(x)

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, conv_func, archws, archas, in_channels, **kwargs):
        super(InceptionB, self).__init__()
        assert len(archas) == 4
        assert len(archws) == 4
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

        self.branch3x3 = conv_func(
            in_channels, 384, archws[0], archas[0], bias=False, kernel_size=3, stride=2, **kwargs)

        self.branch3x3dbl_1 = conv_func(
            in_channels, 64, archws[1], archas[1], bias=False, kernel_size=1, **kwargs)
        self.branch3x3dbl_2 = BasicQuantConv2d(
            conv_func, 64, 96, archws[2], archas[2], kernel_size=3, padding=1, **kwargs)
        self.branch3x3dbl_3 = BasicQuantConv2d(
            conv_func, 96, 96, archws[3], archas[3], kernel_size=3, stride=2, **kwargs)

    def forward(self, x):
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.bn(x)

        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, conv_func, archws, archas, in_channels, channels_7x7, **kwargs):
        super(InceptionC, self).__init__()
        assert len(archas) == 10
        assert len(archws) == 10
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

        self.branch1x1 = conv_func(
            in_channels, 192, archws[0], archas[0], bias=False, kernel_size=1, **kwargs)

        c7 = channels_7x7
        self.branch7x7_1 = conv_func(
            in_channels, c7, archws[1], archas[1], bias=False, kernel_size=1, **kwargs)
        self.branch7x7_2 = BasicQuantConv2d(
            conv_func, c7, c7, archws[2], archas[2], kernel_size=(1, 7), padding=(0, 3), **kwargs)
        self.branch7x7_3 = BasicQuantConv2d(
            conv_func, c7, 192, archws[3], archas[3], kernel_size=(7, 1), padding=(3, 0), **kwargs)

        self.branch7x7dbl_1 = conv_func(
            in_channels, c7, archws[4], archas[4], bias=False, kernel_size=1, **kwargs)
        self.branch7x7dbl_2 = BasicQuantConv2d(
            conv_func, c7, c7, archws[5], archas[5], kernel_size=(7, 1), padding=(3, 0), **kwargs)
        self.branch7x7dbl_3 = BasicQuantConv2d(
            conv_func, c7, c7, archws[6], archas[6], kernel_size=(1, 7), padding=(0, 3), **kwargs)
        self.branch7x7dbl_4 = BasicQuantConv2d(
            conv_func, c7, c7, archws[7], archas[7], kernel_size=(7, 1), padding=(3, 0), **kwargs)
        self.branch7x7dbl_5 = BasicQuantConv2d(
            conv_func, c7, 192, archws[8], archas[8], kernel_size=(1, 7), padding=(0, 3), **kwargs)

        self.branch_pool = BasicQuantConv2d(
            conv_func, in_channels, 192, archws[9], archas[9], kernel_size=1, **kwargs)

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        x = self.bn(x)

        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, conv_func, archws, archas, in_channels, **kwargs):
        super(InceptionD, self).__init__()
        assert len(archas) == 6
        assert len(archws) == 6
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

        self.branch3x3_1 = conv_func(
            in_channels, 192, archws[0], archas[0], bias=False, kernel_size=1, **kwargs)
        self.branch3x3_2 = BasicQuantConv2d(
            conv_func, 192, 320, archws[1], archas[1], kernel_size=3, stride=2, **kwargs)

        self.branch7x7x3_1 = conv_func(
            in_channels, 192, archws[2], archas[2], bias=False, kernel_size=1, **kwargs)
        self.branch7x7x3_2 = BasicQuantConv2d(
            conv_func, 192, 192, archws[3], archas[3], kernel_size=(1, 7), padding=(0, 3), **kwargs)
        self.branch7x7x3_3 = BasicQuantConv2d(
            conv_func, 192, 192, archws[4], archas[4], kernel_size=(7, 1), padding=(3, 0), **kwargs)
        self.branch7x7x3_4 = BasicQuantConv2d(
            conv_func, 192, 192, archws[5], archas[5], kernel_size=3, stride=2, **kwargs)

    def forward(self, x):
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.bn(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, conv_func, archws, archas, in_channels, **kwargs):
        super(InceptionE, self).__init__()
        assert len(archas) == 9
        assert len(archws) == 9
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001)

        self.branch1x1 = conv_func(
            in_channels, 320, archws[0], archas[0], bias=False, kernel_size=1, **kwargs)

        self.branch3x3_1 = conv_func(
            in_channels, 384, archws[1], archas[1], bias=False, kernel_size=1, **kwargs)
        self.branch3x3_bn = nn.BatchNorm2d(384, eps=0.001)
        self.branch3x3_2a = conv_func(
            384, 384, archws[2], archas[2], bias=False, kernel_size=(1, 3), padding=(0, 1), **kwargs)
        self.branch3x3_2b = conv_func(
            384, 384, archws[3], archas[3], bias=False, kernel_size=(3, 1), padding=(1, 0), **kwargs)

        self.branch3x3dbl_1 = conv_func(
            in_channels, 448, archws[4], archas[4], bias=False, kernel_size=1, **kwargs)
        self.branch3x3dbl_2 = BasicQuantConv2d(
            conv_func, 448, 384, archws[5], archas[5], kernel_size=3, padding=1, **kwargs)
        self.branch3x3dbl_bn = nn.BatchNorm2d(384, eps=0.001)
        self.branch3x3dbl_3a = conv_func(
            384, 384, archws[6], archas[6], bias=False, kernel_size=(1, 3), padding=(0, 1), **kwargs)
        self.branch3x3dbl_3b = conv_func(
            384, 384, archws[7], archas[7], bias=False, kernel_size=(3, 1), padding=(1, 0), **kwargs)

        self.branch_pool = BasicQuantConv2d(
            conv_func, in_channels, 192, archws[8], archas[8], kernel_size=1, **kwargs)

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        x = self.bn(x)

        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_bn(branch3x3)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_bn(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class Inception3(nn.Module):

    def __init__(self, conv_func, archws, archas, num_classes=1000, aux_logits=False, **kwargs):
        super(Inception3, self).__init__()
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))
        assert len(archas) == 93
        assert len(archws) == 93
        self.conv_func = conv_func
        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = nn.Conv2d(3, 32, bias=False, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicQuantConv2d(
            conv_func, 32, 32, archws[0], archas[0], kernel_size=3, **kwargs)
        self.Conv2d_2b_3x3 = BasicQuantConv2d(
            conv_func, 32, 64, archws[1], archas[1], kernel_size=3, padding=1, **kwargs)
        self.Conv2d_3b_1x1 = BasicQuantConv2d(
            conv_func, 64, 80, archws[2], archas[2], kernel_size=1, **kwargs)
        self.Conv2d_4a_3x3 = BasicQuantConv2d(
            conv_func, 80, 192, archws[3], archas[3], kernel_size=3, **kwargs)
        self.Mixed_5b = InceptionA(
            conv_func, archws[4:11], archas[4:11], 192, pool_features=32, **kwargs)
        self.Mixed_5c = InceptionA(
            conv_func, archws[11:18], archas[11:18], 256, pool_features=64, **kwargs)
        self.Mixed_5d = InceptionA(
            conv_func, archws[18:25], archas[18:25], 288, pool_features=64, **kwargs)
        self.Mixed_6a = InceptionB(
            conv_func, archws[25:29], archas[25:29], 288, **kwargs)
        self.Mixed_6b = InceptionC(
            conv_func, archws[29:39], archas[29:39], 768, channels_7x7=128, **kwargs)
        self.Mixed_6c = InceptionC(
            conv_func, archws[39:49], archas[39:49], 768, channels_7x7=160, **kwargs)
        self.Mixed_6d = InceptionC(
            conv_func, archws[49:59], archas[49:59], 768, channels_7x7=160, **kwargs)
        self.Mixed_6e = InceptionC(
            conv_func, archws[59:69], archas[59:69], 768, channels_7x7=192, **kwargs)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(conv_func, archws[69:75], archas[69:75], 768, **kwargs)
        self.Mixed_7b = InceptionE(conv_func, archws[75:84], archas[75:84], 1280, **kwargs)
        self.Mixed_7c = InceptionE(conv_func, archws[84:], archas[84:], 2048, **kwargs)
        self.Mixed_7c_bn = nn.BatchNorm2d(2048, eps=0.001)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        x = self.Mixed_7c_bn(x)
        x = F.relu(x, inplace=True)
        # 8 x 8 x 2048
        assert x.shape[2] == 8
        assert x.shape[3] == 8
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
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


def quantinception_2w2a(arch_cfg_path, **kwargs):
    archas, archws = [2] * 93, [2] * 93
    assert len(archas) == 93
    assert len(archws) == 93
    return Inception3(qm.QuantActivConv2d, archws, archas, **kwargs)


def quantinception_pretrained_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    best_activ = [2, 1, 2, 0, 2, 2, 1, 2, 1, 0, 1, 2, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 1, 0, 1,
                  0, 1, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                  2, 2, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    best_weight = [0, 0, 3, 0, 3, 2, 0, 2, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 3, 3, 1, 2, 0, 0, 2,
                   0, 2, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 1, 1, 0, 0, 0, 0, 0, 2, 3,
                   0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
                   3, 3, 2, 2, 3, 0, 0, 2, 3, 2, 2, 3, 3, 2, 0, 2, 2, 3]
    archas = [abits[a] for a in best_activ]
    archws = [wbits[w] for w in best_weight]
    assert len(archas) == 93
    assert len(archws) == 93
    return Inception3(qm.QuantActivConv2d, archws, archas, **kwargs)


def quantinception_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 93
    assert len(archws) == 93
    return Inception3(qm.QuantActivConv2d, archws, archas, **kwargs)
