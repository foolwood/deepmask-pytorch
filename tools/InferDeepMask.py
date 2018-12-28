import time
import torch
import torch.nn as nn
import numpy as np
import cv2

import torchvision.utils as vutils  # visualization
import matplotlib.pyplot as plt  # visualization


class Infer(object):
    def __init__(self, model, nps=500, scales=(1.,),
                 meanstd={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                 iSz=160, device='cpu', timer=True):
        self.trunk = model.trunk
        self.mHead = model.maskBranch
        self.sHead = model.scoreBranch
        self.nps = nps
        self.mean = torch.from_numpy(np.array(meanstd['mean']).astype(np.float32)).view(1, 3, 1, 1).to(device)
        self.std = torch.from_numpy(np.array(meanstd['std']).astype(np.float32)).view(1, 3, 1, 1).to(device)
        self.iSz, self.bw = iSz, iSz // 2
        self.device = device
        self.timer = np.zeros(6)
        self.display_time = timer

        self.scales = scales
        self.pyramid = [nn.UpsamplingBilinear2d(scale_factor=s).to(device) for s in self.scales]

    def forward(self, img):
        tic = time.time()
        imgPyramid = [pyramid(img) for pyramid in self.pyramid]
        self.timer[0] = time.time() - tic

        self.mask, self.score = [], []
        for inp in imgPyramid:
            tic = time.time()
            imgPad = nn.ConstantPad2d(self.bw, 0.5).cuda()(inp)
            # cv2.imshow('pad image', np.transpose(imgPad.squeeze().cpu().data.numpy(), axes=(1, 2, 0))[:,::-1])
            # cv2.waitKey(0)
            imgPad = imgPad.sub_(self.mean).div_(self.std)
            self.timer[1] += time.time() - tic

            tic = time.time()
            outTrunk = self.trunk(imgPad)
            self.timer[2] += time.time() - tic

            tic = time.time()
            outMask = self.mHead(outTrunk)
            self.timer[3] += time.time() - tic

            tic = time.time()
            outScore = self.sHead(outTrunk)
            self.timer[4] += time.time() - tic

            # mask_show = vutils.make_grid(outMask.sigmoid().transpose(0, 1), nrow=outScore.shape[-1], pad_value=0)
            # mask_show_numpy = np.transpose(mask_show.cpu().data.numpy(), axes=(1, 2, 0))
            # plt.imshow(mask_show_numpy[:,:,0], cmap='jet')
            # plt.show()
            self.mask.append(outMask.sigmoid().cpu().data.numpy())
            self.score.append(outScore.sigmoid().cpu().data.numpy())

    def getTopScores(self):
        li = []
        for i, sc in enumerate(self.score):
            h, w = sc.shape[2:]
            sc = sc.flatten()
            sIds, sS = zip(*sorted(enumerate(sc), key=lambda a: a[1], reverse=True))
            for ss, sid in zip(sS, sIds):
                li.append([ss, i, sid, sid//w, sid % w])
        li = sorted(li, key=lambda l: l[0], reverse=True)
        topScores = li[:self.nps]
        self.topScores = topScores

    def crop_hwc(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0]-1) / bbox[2]
        b = (out_sz[1]-1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def getTopMasks(self, thr, h, w):
        masks, ts, nps = self.mask, self.topScores, self.nps
        # thr = np.log(thr / (1 - thr))
        topMasks = np.zeros((h, w, nps), dtype=np.uint8)
        topScores = np.zeros(nps)
        for i in range(nps):
            scale, sid, x, y = ts[i][1:]
            s = self.scales[scale]
            mask = masks[scale][0, sid]
            mask = cv2.resize(mask, (self.iSz, self.iSz))
            # cv2.imshow('mask', mask)
            # cv2.waitKey(0)
            imgMask = self.crop_hwc(mask, (self.bw - 16*y, self.bw - 16*x, w*s, h*s), (w, h))
            imgMask = imgMask > thr

            topMasks[:, :, i] = imgMask.copy()
            topScores[i] = ts[i][0]
        return topMasks, topScores

    def getTopProps(self, thr, h, w):
        tic = time.time()

        self.getTopScores()
        topMasks, topScores = self.getTopMasks(thr, h, w)

        self.timer[5] = time.time() - tic
        if self.display_time:
            self.printTiming()
        return topMasks, topScores

    def printTiming(self):
        t = self.timer
        print('| time pyramid: {:.1f} ms'.format(t[0]*1000))
        print('| time pre-process: {:.1f} ms'.format(t[1]*1000))
        print('| time trunk: {:.1f} ms'.format(t[2]*1000))
        print('| time mask branch: {:.1f} ms'.format(t[3]*1000))
        print('| time score branch: {:.1f} ms'.format(t[4]*1000))
        print('| time post processing: {:.1f} ms'.format(t[5]*1000))
        print('| time total: {:.1f} ms'.format(sum(t)*1000))
