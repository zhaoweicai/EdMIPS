from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

gaussian_steps = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.336}
hwgq_steps = {1: 0.799, 2: 0.538, 3: 0.3217, 4: 0.185}


class _gauss_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        alpha = x.std().item()
        step *= alpha
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _gauss_quantize_resclaed_step(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _hwgq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step):
        y = torch.round(x / step) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HWGQ(nn.Module):
    def __init__(self, bit=2):
        super(HWGQ, self).__init__()
        self.bit = bit
        if bit < 32:
            self.step = hwgq_steps[bit]
        else:
            self.step = None

    def forward(self, x):
        if self.bit >= 32:
            return x.clamp(min=0.0)
        lvls = float(2 ** self.bit - 1)
        clip_thr = self.step * lvls
        y = x.clamp(min=0.0, max=clip_thr)
        return _hwgq.apply(y, self.step)


class QuantConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        self.bit = kwargs.pop('bit', 1)
        super(QuantConv2d, self).__init__(*kargs, **kwargs)
        assert self.bit > 0
        self.step = gaussian_steps[self.bit]

    def forward(self, input):
        # quantized conv, otherwise regular
        if self.bit < 32:
            assert self.bias is None
            quant_weight = _gauss_quantize.apply(self.weight, self.step, self.bit)
            out = F.conv2d(
                input, quant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            out = F.conv2d(
                input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class QuantLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        self.bit = kwargs.pop('bit', 1)
        super(QuantLinear, self).__init__(*kargs, **kwargs)
        assert self.bit > 0
        self.step = gaussian_steps[self.bit]

    def forward(self, input):
        # quantized linear, otherwise regular
        if self.bit < 32:
            assert self.bias is None
            quant_weight = _gauss_quantize.apply(self.weight, self.step, self.bit)
            out = F.linear(input, quant_weight, self.bias)
        else:
            out = F.linear(input, self.weight, self.bias)
        return out


class QuantActivConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbit=1, abit=2, **kwargs):
        super(QuantActivConv2d, self).__init__()
        self.abit = abit
        self.wbit = wbit
        self.activ = HWGQ(abit)
        self.conv = QuantConv2d(inplane, outplane, bit=wbit, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        out = self.activ(input)
        out = self.conv(out)
        return out


class QuantActivLinear(nn.Module):

    def __init__(self, inplane, outplane, wbit=1, abit=2, **kwargs):
        super(QuantActivLinear, self).__init__()
        self.abit = abit
        self.wbit = wbit
        self.activ = HWGQ(abit)
        self.linear = QuantLinear(inplane, outplane, bit=wbit, **kwargs)
        # complexities
        self.param_size = inplane * outplane * 1e-6
        self.register_buffer('size_product', torch.tensor(self.param_size, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        tmp = torch.tensor(input.shape[1] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        out = self.activ(input)
        out = self.linear(out)
        return out


class MixQuantActiv(nn.Module):

    def __init__(self, bits):
        super(MixQuantActiv, self).__init__()
        self.bits = bits
        self.alpha_activ = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_activ.data.fill_(0.01)
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            self.mix_activ.append(HWGQ(bit=bit))

    def forward(self, input):
        outs = []
        sw = F.softmax(self.alpha_activ, dim=0)
        for i, branch in enumerate(self.mix_activ):
            outs.append(branch(input) * sw[i])
        activ = sum(outs)
        return activ


class MixQuantConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(MixQuantConv2d, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.conv_list = nn.ModuleList()
        self.steps = []
        for bit in self.bits:
            assert 0 < bit < 32
            self.conv_list.append(nn.Conv2d(inplane, outplane, **kwargs))
            self.steps.append(gaussian_steps[bit])

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        for i, bit in enumerate(self.bits):
            weight = self.conv_list[i].weight
            weight_std = weight.std().item()
            step = self.steps[i] * weight_std
            quant_weight = _gauss_quantize_resclaed_step.apply(weight, step, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        conv = self.conv_list[0]
        out = F.conv2d(
            input, mix_quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


class SharedMixQuantConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(SharedMixQuantConv2d, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.conv = nn.Conv2d(inplane, outplane, **kwargs)
        self.steps = []
        for bit in self.bits:
            assert 0 < bit < 32
            self.steps.append(gaussian_steps[bit])

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        conv = self.conv
        weight = conv.weight
        # save repeated std computation for shared weights
        weight_std = weight.std().item()
        for i, bit in enumerate(self.bits):
            step = self.steps[i] * weight_std
            quant_weight = _gauss_quantize_resclaed_step.apply(weight, step, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        out = F.conv2d(
            input, mix_quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


class MixActivConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight=False, **kwargs):
        super(MixActivConv2d, self).__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        # build mix-precision branches
        self.mix_activ = MixQuantActiv(self.abits)
        self.share_weight = share_weight
        if share_weight:
            self.mix_weight = SharedMixQuantConv2d(inplane, outplane, self.wbits, **kwargs)
        else:
            self.mix_weight = MixQuantConv2d(inplane, outplane, self.wbits, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        out = self.mix_activ(input)
        out = self.mix_weight(out)
        return out

    def complexity_loss(self):
        sw = F.softmax(self.mix_activ.alpha_activ, dim=0)
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += sw[i] * abits[i]
        sw = F.softmax(self.mix_weight.alpha_weight, dim=0)
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += sw[i] * wbits[i]
        complexity = self.size_product.item() * mix_abit * mix_wbit
        return complexity

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()
        best_activ = prob_activ.argmax()
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += prob_activ[i] * abits[i]
        prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
        prob_weight = prob_weight.detach().cpu().numpy()
        best_weight = prob_weight.argmax()
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += prob_weight[i] * wbits[i]
        if self.share_weight:
            weight_shape = list(self.mix_weight.conv.weight.shape)
        else:
            weight_shape = list(self.mix_weight.conv_list[0].weight.shape)
        print('idx {} with shape {}, activ alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, mix_wbit, memory_size, mix_abit))
        print('idx {} with shape {}, weight alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'param: {:.3f}M * {:.3f}'.format(layer_idx, weight_shape, prob_weight, size_product,
                                               mix_abit, mix_wbit, self.param_size, mix_wbit))
        best_arch = {'best_activ': [best_activ], 'best_weight': [best_weight]}
        bitops = size_product * abits[best_activ] * wbits[best_weight]
        bita = memory_size * abits[best_activ]
        bitw = self.param_size * wbits[best_weight]
        mixbitops = size_product * mix_abit * mix_wbit
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * mix_wbit
        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw


class SharedMixQuantLinear(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(SharedMixQuantLinear, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.linear = nn.Linear(inplane, outplane, **kwargs)
        self.steps = []
        for bit in self.bits:
            assert 0 < bit < 32
            self.steps.append(gaussian_steps[bit])

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        linear = self.linear
        weight = linear.weight
        # save repeated std computation for shared weights
        weight_std = weight.std().item()
        for i, bit in enumerate(self.bits):
            step = self.steps[i] * weight_std
            quant_weight = _gauss_quantize_resclaed_step.apply(weight, step, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        out = F.linear(input, mix_quant_weight, linear.bias)
        return out


class MixActivLinear(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight=True, **kwargs):
        super(MixActivLinear, self).__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        # build mix-precision branches
        self.mix_activ = MixQuantActiv(self.abits)
        assert share_weight
        self.share_weight = share_weight
        self.mix_weight = SharedMixQuantLinear(inplane, outplane, self.wbits, **kwargs)
        # complexities
        self.param_size = inplane * outplane * 1e-6
        self.register_buffer('size_product', torch.tensor(self.param_size, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        tmp = torch.tensor(input.shape[1] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        out = self.mix_activ(input)
        out = self.mix_weight(out)
        return out

    def complexity_loss(self):
        sw = F.softmax(self.mix_activ.alpha_activ, dim=0)
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += sw[i] * abits[i]
        sw = F.softmax(self.mix_weight.alpha_weight, dim=0)
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += sw[i] * wbits[i]
        complexity = self.size_product.item() * mix_abit * mix_wbit
        return complexity

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()
        best_activ = prob_activ.argmax()
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += prob_activ[i] * abits[i]
        prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
        prob_weight = prob_weight.detach().cpu().numpy()
        best_weight = prob_weight.argmax()
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += prob_weight[i] * wbits[i]
        weight_shape = list(self.mix_weight.linear.weight.shape)
        print('idx {} with shape {}, activ alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, mix_wbit, memory_size, mix_abit))
        print('idx {} with shape {}, weight alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'param: {:.3f}M * {:.3f}'.format(layer_idx, weight_shape, prob_weight, size_product,
                                               mix_abit, mix_wbit, self.param_size, mix_wbit))
        best_arch = {'best_activ': [best_activ], 'best_weight': [best_weight]}
        bitops = size_product * abits[best_activ] * wbits[best_weight]
        bita = memory_size * abits[best_activ]
        bitw = self.param_size * wbits[best_weight]
        mixbitops = size_product * mix_abit * mix_wbit
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * mix_wbit
        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw
