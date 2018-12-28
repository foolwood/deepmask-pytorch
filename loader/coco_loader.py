from os.path import join
import numpy as np
import random
from collections import namedtuple
import cv2
from PIL import Image
from torch.utils import data
from loader.pycocotools.coco import COCO

_Config = namedtuple('Config', ['datadir', 'iSz', 'gSz',
                                 'scale', 'shift', 'maxload',
                                 'testmaxload', 'batch', 'hfreq'])
_config = _Config(datadir='/media/torrvision/Data/coco',
          iSz=160, gSz=112, scale=.25, shift=16,
          maxload=4000, testmaxload=500, batch=32,
          hfreq=0.5)


def range_end(start, stop, step=1):
    return np.arange(start, stop+step, step)


class cocoDataset(data.Dataset):
    """cocoDataset

    http://cocodataset.org

    Data is derived from COCO17, and can be downloaded from here:
    http://cocodataset.org/#download
    """
    def __init__(self, config=None, split='val'):
        self.datadir = config.datadir if hasattr(config, 'datadir') else join('data', 'coco')
        self.split = split
        self.annFile = join(self.datadir, 'annotations/instances_%s2017.json' % split)
        self.coco = COCO(self.annFile)

        self.mean = np.array([0.485, 0.456, 0.406]).astype(np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).astype(np.float32).reshape(3, 1, 1)

        self.iSz = config.iSz
        self.objSz = np.ceil(config.iSz * 128 / 224)
        self.wSz = config.iSz + 32
        self.gSz = config.gSz
        self.scale = config.scale
        self.shift = config.shift
        self.hfreq = config.hfreq

        self.imgIds = self.coco.getImgIds()
        self.annIds = self.coco.getAnnIds()
        self.catIds = self.coco.getCatIds()
        self.nImages = len(self.imgIds)
        self.batch = config.batch

        if split == 'train':
            self.__size = config.maxload * config.batch
        elif split == 'val':
            self.__size = config.testmaxload * config.batch

        self.rand_list = np.random.choice([0, 1], size=self.__size // self.batch,
                                          p=[self.hfreq, 1-self.hfreq])

        if config.hfreq < 1:
            self.scales = range_end(-3, 2, .25)
            self.createBBstruct(self.objSz, config.scale)

    def __len__(self):
        return self.__size

    def __getitem__(self, index):
        """Get a patch, label, and status.
        Returns:
            img: Image patch (b*3*H*W numpy.ndarray [0 ~ 1]).
            label: Image label (b*1*gSz*gSz /b*1*1*1 numpy.ndarray [-1, 1]).
            head_status: head status (0 for mask, 1 for score).
        """
        head_status = self.rand_list[index // self.batch]
        if head_status == 0:
            img, label = self.maskSampling()
        elif head_status == 1:
            img, label = self.scoreSampling()

        if random.random() > .5:
            img = img[:, :, ::-1].copy()
            if head_status == 0:
                label = label[:, :, ::-1].copy()

        img = (img-self.mean)/self.std
        return img.astype(np.float32), \
               label.astype(np.float32), head_status

    def shuffle(self):
        self.rand_list = np.random.choice([0, 1], size=self.__size // self.batch,
                                          p=[self.hfreq, 1-self.hfreq])

    def createBBstruct(self, objSz, scale):
        bbStruct = []
        for imId in self.imgIds:
            annIds = self.coco.getAnnIds(imgIds=imId)
            bbs = {'scales': []}
            for annId in annIds:
                ann = self.coco.loadAnns(annId)[0]
                bbGt = ann['bbox']
                x0, y0, w, h = bbGt
                xc, yc, maxDim = x0 + w / 2, y0 + h / 2, max(w, h)
                for s in range_end(-32., 32.):
                    d1 = objSz * 2 ** ((s - 1) * scale)
                    d2 = objSz * 2 ** ((s + 1) * scale)
                    if d1 < maxDim <= d2:
                        ss = -s * scale
                        xcS, ycS = xc * (2 ** ss), yc * (2 ** ss)
                        sss = str(ss).replace('.', '_')
                        if sss not in bbs:
                            bbs[sss] = []
                            bbs['scales'].append(sss)
                        bbs[sss].append((xcS, ycS))
                        break
            bbStruct.append(bbs)
        self.bbStruct = bbStruct

    def maskSampling(self):
        iSz, wSz, gSz = self.iSz, self.wSz, self.gSz
        catId = random.choice(self.catIds)
        annIds = self.coco.getAnnIds(catIds=catId)
        ann = None
        while not ann or ann['iscrowd'] == 1 or ann['area'] < 100 or \
                ann['bbox'][2] < 5 or ann['bbox'][3] < 5:
            ann = self.coco.loadAnns(random.choice(annIds))[0]

        bbox = self.jitterBox(ann['bbox'])
        imgName = self.coco.loadImgs(ann['image_id'])[0]['file_name']
        imgPath = '%s/%s2017/%s' % (self.datadir, self.split, imgName)

        img = np.array(Image.open(imgPath).convert('RGB'), dtype=np.float32)/255.
        img = self.crop_hwc(img, bbox, wSz, (0.5, 0.5, 0.5))

        iSzR = iSz * (bbox[3] / wSz)
        xc, yc = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        bboxInpSz = [xc - iSzR / 2, yc - iSzR / 2, iSzR, iSzR]
        mo = self.coco.annToMask(ann).astype(np.float32)  # float for bilinear interpolation
        lbl = self.crop_hwc(mo, bboxInpSz, gSz) > 0.5
        lbl = lbl * 2 - 1

        img = np.transpose(img, (2, 0, 1))  # 3*H*W
        lbl = np.expand_dims(lbl, axis=0)   # 1*H*W
        return img, lbl

    def scoreSampling(self):
        while True:
            idx = random.randint(0, self.nImages-1)
            bb = self.bbStruct[idx]
            if len(bb['scales']) != 0:  # have object
                break

        imgId = self.imgIds[idx]
        imgName = self.coco.loadImgs(imgId)[0]['file_name']
        imgPath = '%s/%s2017/%s' % (self.datadir, self.split, imgName)

        img = np.array(Image.open(imgPath).convert('RGB'), dtype=np.float32)/255.

        h, w = img.shape[:2]

        if random.random() > 0.5:
            x, y, scale = self.posSamplingBB(bb)
            lbl = 1
        else:
            x, y, scale = self.negSamplingBB(bb, w, h)
            lbl = -1
        s = 2 ** -scale
        x, y = min(max(x * s, 0), w), min(max(y * s, 0), h)
        isz = max(self.wSz * s, 10)
        inp = self.crop_hwc(img, [x-isz/2, y-isz/2, isz, isz], self.wSz, (0.5, 0.5, 0.5))

        inp = np.transpose(inp, (2, 0, 1))  # 3*H*W
        lbl = np.reshape(lbl, (1, 1, 1))    # 1*1*1
        return inp, lbl

    def jitterBox(self, box):
        x, y, w, h = box
        xc, yc = x + w / 2, y + h / 2
        maxDim = max(w, h)
        scale = np.log2(maxDim / self.objSz)
        s = scale + random.uniform(-self.scale, self.scale)
        xc = xc + random.uniform(-self.shift, self.shift) * 2 ** s
        yc = yc + random.uniform(-self.shift, self.shift) * 2 ** s
        w, h = self.wSz * 2 ** s, self.wSz * 2 ** s
        return [xc - w / 2, yc - h / 2, w, h]

    def crop_hwc(self, image, bbox, out_sz, padding=0):
        a = (out_sz-1) / bbox[2]
        b = (out_sz-1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def posSamplingBB(self, bb):
        scale = random.choice(bb['scales'])
        x, y = random.choice(bb[scale])
        return x, y, float(scale.replace('_', '.'))

    def negSamplingBB(self, bb, w0, h0):
        neg_flag, c = False, 0
        while not neg_flag and c < 100:
            scale = random.choice(self.scales)
            x, y = random.uniform(0, w0*2**scale), random.uniform(0, h0*2**scale)
            neg_flag = True
            for s in range_end(-10, 10):
                ss = scale + s*self.scale
                sss = str(ss).replace('.', '_')
                if sss in bb:
                    for cc in bb[sss]:
                        dist = np.sqrt((x - cc[0])**2 + (y - cc[1])**2)
                        if dist < 3*self.shift:
                            neg_flag = False
                            break
                if not neg_flag:
                    break
            c += 1
        return x, y, scale


if __name__ == '__main__':
    import time
    import torch
    import matplotlib.pyplot as plt  # visualization
    import torchvision.utils as vutils  # visualization
    from tools.train import BinaryMeter, IouMeter
    visual = False

    dst = cocoDataset(_config)
    bs = _config.batch
    train_loader = data.DataLoader(dst, batch_size=bs, num_workers=0)

    score_meter = BinaryMeter()
    score_meter.reset()
    mask_meter = IouMeter(0.5, len(train_loader.dataset))
    mask_meter.reset()

    tic = time.time()
    for i, data in enumerate(train_loader):
        imgs, labels, head_status = data
        print(i, imgs.shape, labels.shape, head_status.shape)

        if head_status[0] == 0:
            outputs = torch.flip(labels, [0, ])
            mask_meter.add(outputs, labels)
        else:
            outputs = torch.flip(labels, [0, ])
            score_meter.add(outputs, labels)

        if visual:
            img_show = vutils.make_grid(imgs, normalize=True, scale_each=True)
            img_show_numpy = np.transpose(img_show.cpu().data.numpy(), axes=(1, 2, 0))

            iSz_res = torch.nn.functional.interpolate(labels, size=(_config.iSz, _config.iSz))
            pad_res = torch.nn.functional.pad(iSz_res, (16, 16, 16, 16))
            mask_show = vutils.make_grid(pad_res, scale_each=True)
            mask_show_numpy = np.transpose(mask_show.cpu().data.numpy(), axes=(1, 2, 0))

            plt.imshow(img_show_numpy)
            plt.imshow(mask_show_numpy[:, :, 0] > 0, alpha=.7, cmap='jet')
            plt.axis('off')
            plt.subplots_adjust(.05, .05, .95, .95)
            plt.show()
            plt.close()

    print(' IoU: mean %05.2f median %05.2f suc@.5 %05.2f suc@.7 %05.2f | acc %05.2f' % (
        mask_meter.value('mean'), mask_meter.value('median'), mask_meter.value('0.5'), mask_meter.value('0.7'),
        score_meter.value()))
    toc = time.time() - tic
    print('Time used in 1 epoch: {:.2f}s'.format(toc))

