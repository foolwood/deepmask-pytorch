import torch
import torch.nn as nn
import torchvision
from collections import namedtuple

Config = namedtuple('Config', ['iSz', 'oSz', 'gSz'])
default_config = Config(iSz=160, oSz=56, gSz=112)


class Reshape(nn.Module):
    def __init__(self, oSz):
        super(Reshape, self).__init__()
        self.oSz = oSz

    def forward(self, x):
        b = x.shape[0]
        return x.permute(0, 2, 3, 1).view(b, -1, self.oSz, self.oSz)


class SymmetricPad2d(nn.Module):
    def __init__(self, padding):
        super(SymmetricPad2d, self).__init__()
        self.padding = padding
        try:
            self.pad_l, self.pad_b, self.pad_r, self.pad_t = padding
        except:
            self.pad_l, self.pad_b, self.pad_r, self.pad_t = [padding,]*4

    def forward(self, input):
        assert len(input.shape) == 4, "only Dimension=4 implemented"
        h = input.shape[2] + self.pad_t + self.pad_b
        w = input.shape[3] + self.pad_l + self.pad_r
        assert w >= 1 and h >= 1, "input is too small"
        output = torch.zeros(input.shape[0], input.shape[1], h, w).to(input.device)
        c_input = input
        if self.pad_t < 0:
            c_input = c_input.narrow(2, -self.pad_t, c_input.shape[2] + self.pad_t)
        if self.pad_b < 0:
            c_input = c_input.narrow(2, 0, c_input.shape[2] + self.pad_b)
        if self.pad_l < 0:
            c_input = c_input.narrow(3, -self.pad_l, c_input.shape[3] + self.pad_l)
        if self.pad_r < 0:
            c_input = c_input.narrow(3, 0, c_input.shape[3] + self.pad_r)

        c_output = output
        if self.pad_t > 0:
            c_output = c_output.narrow(2, self.pad_t, c_output.shape[2] - self.pad_t)
        if self.pad_b > 0:
            c_output = c_output.narrow(2, 0, c_output.shape[2] - self.pad_b)
        if self.pad_l > 0:
            c_output = c_output.narrow(3, self.pad_l, c_output.shape[3] - self.pad_l)
        if self.pad_r > 0:
            c_output = c_output.narrow(3, 0, c_output.shape[3] - self.pad_r)

        c_output.copy_(c_input)

        assert w >= 2*self.pad_l and w >= 2*self.pad_r and h >= 2*self.pad_t and h >= 2*self.pad_b
        "input is too small"
        for i in range(self.pad_t):
            output.narrow(2, self.pad_t-i-1, 1).copy_(output.narrow(2, self.pad_t+i, 1))
        for i in range(self.pad_b):
            output.narrow(2, output.shape[2] - self.pad_b + i, 1).copy_(
                output.narrow(2, output.shape[2] - self.pad_b - i-1, 1))
        for i in range(self.pad_l):
            output.narrow(3, self.pad_l-i-1, 1).copy_(output.narrow(3, self.pad_l+i, 1))
        for i in range(self.pad_r):
            output.narrow(3, output.shape[3] - self.pad_r + i, 1).copy_(
                output.narrow(3, output.shape[3] - self.pad_r - i-1, 1))
        return output


def updatePadding(net, nn_padding):
    typename = torch.typename(net)
    # print(typename)
    if typename.find('Sequential') >= 0 or typename.find('Bottleneck') >= 0:
        modules_keys = list(net._modules.keys())
        for i in reversed(range(len(modules_keys))):
            subnet = net._modules[modules_keys[i]]
            out = updatePadding(subnet, nn_padding)
            if out != -1:
                p = out
                in_c, out_c, k, s, _, d, g, b = \
                    subnet.in_channels, subnet.out_channels, \
                    subnet.kernel_size[0], subnet.stride[0], \
                    subnet.padding[0], subnet.dilation[0], \
                    subnet.groups, subnet.bias,
                conv_temple = nn.Conv2d(in_c, out_c, k, stride=s, padding=0,
                                        dilation=d, groups=g, bias=b)
                conv_temple.weight = subnet.weight
                conv_temple.bias = subnet.bias
                if p > 1:
                    net._modules[modules_keys[i]] = nn.Sequential(SymmetricPad2d(p), conv_temple)
                else:
                    net._modules[modules_keys[i]] = nn.Sequential(nn_padding(p), conv_temple)
    else:
        if typename.find('torch.nn.modules.conv.Conv2d') >= 0:
            k_sz, p_sz = net.kernel_size[0], net.padding[0]
            if ((k_sz == 3) or (k_sz == 7)) and p_sz != 0:
                return p_sz
    return -1


class DeepMask(nn.Module):
    def __init__(self, config=default_config, context=True):
        super(DeepMask, self).__init__()
        self.config = config
        self.context = context  # without context
        self.strides = 16
        self.fSz = -(-self.config.iSz // self.strides)  # ceil div
        self.trunk = self.creatTrunk()
        updatePadding(self.trunk, nn.ReflectionPad2d)
        self.crop_trick = nn.ZeroPad2d(-16//self.strides)  # for training
        self.maskBranch = self.createMaskBranch()
        self.scoreBranch = self.createScoreBranch()

        npt = sum(p.numel() for p in self.trunk.parameters()) / 1e+06
        npm = sum(p.numel() for p in self.maskBranch.parameters()) / 1e+06
        nps = sum(p.numel() for p in self.scoreBranch.parameters()) / 1e+06
        print('| number of paramaters trunk: {:.3f} M'.format(npt))
        print('| number of paramaters mask branch: {:.3f} M'.format(npm))
        print('| number of paramaters score branch: {:.3f} M'.format(nps))
        print('| number of paramaters total: {:.3f} M'.format(npt + nps + npm))

    def forward(self, x):
        feat = self.trunk(x)
        if self.context:
            feat = self.crop_trick(feat)
        mask = self.maskBranch(feat)
        score = self.scoreBranch(feat)
        return mask, score

    def creatTrunk(self):
        resnet50 = torchvision.models.resnet50(pretrained=True)
        trunk1 = nn.Sequential(*list(resnet50.children())[:-3])
        trunk2 = nn.Sequential(
            nn.Conv2d(1024, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, self.fSz)
        )
        return nn.Sequential(trunk1, trunk2)

    def createMaskBranch(self):
        maskBranch = nn.Sequential(
            nn.Conv2d(512, self.config.oSz**2, 1),
            Reshape(self.config.oSz),
        )
        if self.config.gSz > self.config.oSz:
            upSample = nn.UpsamplingBilinear2d(size=[self.config.gSz, self.config.gSz])
            maskBranch = nn.Sequential(maskBranch, upSample)
        return maskBranch

    def createScoreBranch(self):
        scoreBranch = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 1024, 1),
            nn.Threshold(0, 1e-6),  # do not know why
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1, 1),
        )
        return scoreBranch


if __name__ == '__main__':
    a = SymmetricPad2d(3)
    x = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    out = a(x)
    print(out)
    import torch
    Config = namedtuple('Config', ['iSz', 'oSz', 'gSz'])
    config = Config(iSz=160, oSz=56, gSz=112)
    model = DeepMask(config).cuda()
    # training mode
    x = torch.rand(32, 3, config.iSz+32, config.iSz+32).cuda()
    pred_mask, pred_cls = model(x)
    print("Output (training mode)", pred_mask.shape, pred_cls.shape)

    # full image testing mode
    model.context = False  # really important!!
    input_size = config.iSz + model.strides * 16 + (model.context * 32)
    x = torch.rand(8, 3, input_size, input_size).cuda()
    pred_mask, pred_cls = model(x)
    print("Output (testing mode)", pred_mask.shape, pred_cls.shape)
