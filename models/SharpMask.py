import os
import torch.nn as nn
from collections import namedtuple
from models.DeepMask import DeepMask
from utils.load_helper import load_pretrain

Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'km', 'ks'])
default_config = Config(iSz=160, oSz=56, gSz=160, km=32, ks=32)


class RefineModule(nn.Module):
    def __init__(self, l1, l2, l3):
        super(RefineModule, self).__init__()
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3

    def forward(self, x):
        x1 = self.layer1(x[0])
        x2 = self.layer2(x[1])

        y = x1 + x2
        y = self.layer3(y)
        return y


class SharpMask(nn.Module):
    def __init__(self, config=default_config, context=True):
        super(SharpMask, self).__init__()
        self.context = context  # with context
        self.km, self.ks = config.km, config.ks
        self.skpos = [6, 5, 4, 2]

        deepmask = DeepMask(config)
        deeomask_resume = os.path.join('exps', 'deepmask', 'train', 'model_best.pth.tar')
        assert os.path.exists(deeomask_resume), "Please train DeepMask first"
        deepmask = load_pretrain(deepmask, deeomask_resume)
        self.trunk = deepmask.trunk
        self.crop_trick = deepmask.crop_trick
        self.scoreBranch = deepmask.scoreBranch
        self.maskBranchDM = deepmask.maskBranch
        self.fSz = deepmask.fSz

        self.refs = self.createTopDownRefinement()  # create refinement modules

        nph = sum(p.numel() for h in self.neths for p in h.parameters()) / 1e+06
        npv = sum(p.numel() for h in self.netvs for p in h.parameters()) / 1e+06
        print('| number of paramaters net h: {:.3f} M'.format(nph))
        print('| number of paramaters net v: {:.3f} M'.format(npv))
        print('| number of paramaters total: {:.3f} M'.format(nph + npv))

    def forward(self, x):
        inps = list()
        for i, l in enumerate(self.trunk.children()):
            for j, ll in enumerate(l.children()):
                x = ll(x)
                if i == 0 and j == (len(l)-1) and self.context:
                    x = self.crop_trick(x)
                # print(x.shape)
                if i == 0 and j in self.skpos:
                    inps.append(x)
        # forward refinement modules
        currentOutput = self.refs[0](x)
        for k in range(len(self.refs)-2):
            x_f = inps[-(k+1)]
            currentOutput = self.refs[k+1]((x_f, currentOutput))
        currentOutput = self.refs[-1](currentOutput)
        return currentOutput, self.scoreBranch(x)

    def train(self, mode=True):
        self.training = mode
        if mode:
            for module in self.children():
                module.train(False)
            for module in self.refs.children():
                module.train(mode)
        else:
            for module in self.children():
                module.train(mode)
        return self

    def createHorizontal(self):
        neths = nn.ModuleList()
        nhu1, nhu2, crop = 0, 0, 0
        for i in range(len(self.skpos)):
            h = []
            nInps = self.ks // 2 ** i
            if i == 0:
                nhu1, nhu2, crop = 1024, 64, 0 if self.context else 0
            elif i == 1:
                nhu1, nhu2, crop = 512, 64, -2 if self.context else 0
            elif i == 2:
                nhu1, nhu2, crop = 256, 64, -4 if self.context else 0
            elif i == 3:
                nhu1, nhu2, crop = 64, 64, -8 if self.context else 0
            if crop != 0:
                h.append(nn.ZeroPad2d(crop))
            h.append(nn.ReflectionPad2d(1))
            h.append(nn.Conv2d(nhu1, nhu2, 3))
            h.append(nn.ReLU(inplace=True))

            h.append(nn.ReflectionPad2d(1))
            h.append(nn.Conv2d(nhu2, nInps, 3))
            h.append(nn.ReLU(inplace=True))

            h.append(nn.ReflectionPad2d(1))
            h.append(nn.Conv2d(nInps, nInps // 2, 3))

            neths.append(nn.Sequential(*h))
        return neths

    def createVertical(self):
        netvs = nn.ModuleList()
        netvs.append(nn.ConvTranspose2d(512, self.km, self.fSz))

        for i in range(len(self.skpos)):
            netv = []
            nInps = self.km // 2 ** i
            netv.append(nn.ReflectionPad2d(1))
            netv.append(nn.Conv2d(nInps, nInps, 3))
            netv.append(nn.ReLU(inplace=True))

            netv.append(nn.ReflectionPad2d(1))
            netv.append(nn.Conv2d(nInps, nInps // 2, 3))

            netvs.append(nn.Sequential(*netv))

        return netvs

    def refinement(self, neth, netv):
        return RefineModule(neth, netv,
                            nn.Sequential(nn.ReLU(inplace=True),
                             nn.UpsamplingNearest2d(scale_factor=2)))

    def createTopDownRefinement(self):
        # create horizontal nets
        self.neths = self.createHorizontal()

        # create vertical nets
        self.netvs = self.createVertical()

        refs = nn.ModuleList()
        refs.append(self.netvs[0])
        for i in range(len(self.skpos)):
            refs.append(self.refinement(self.neths[i], self.netvs[i+1]))
        refs.append(nn.Sequential(nn.ReflectionPad2d(1),
                                  nn.Conv2d(self.km // 2 ** (len(refs)-1), 1, 3)))

        return refs


if __name__ == '__main__':
    import torch
    Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'km', 'ks'])
    config = Config(iSz=160, oSz=56, gSz=160, km=32, ks=32)
    model = SharpMask(config).cuda()
    # training mode
    x = torch.rand(32, 3, config.iSz+32, config.iSz+32).cuda()
    pred_mask = model(x, True)
    print("Output (training mode)", pred_mask.shape)

    # full image testing mode
    # x = torch.rand(8, 3, config.iSz + 160, config.iSz + 160).cuda()
    # pred_mask, pred_cls = model(x, train=False)
    # print("Output (testing mode)", pred_mask.shape, pred_cls.shape)
