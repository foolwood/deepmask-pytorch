import argparse
import models
import numpy as np
import time
import json
from os.path import join, isdir
from os import makedirs
import matplotlib.pyplot as plt
from PIL import Image
import torch
from loader.pycocotools.mask import encode
from loader.pycocotools.coco import COCO
from loader.pycocotools.cocoeval import COCOeval
from tools.InferDeepMask import Infer
from utils.load_helper import load_pretrain

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch full scene evaluation of DeepMask/SharpMask')

parser.add_argument('--rundir', default='./exps/', help='experiments directory')
parser.add_argument('--datadir', default='data/coco', help='data directory')
parser.add_argument('--split', default='val', help='dataset split to be used (train/val)')
parser.add_argument('--nps', default=1000, type=int, help='number of proposals')
parser.add_argument('--thr', default=.2, type=float, help='mask binary threshold')
parser.add_argument('--smin', default=-2.5, type=float, help='min scale')
parser.add_argument('--smax', default=1, type=float, help='max scale')
parser.add_argument('--sstep', default=.5, type=float, help='scale step')
parser.add_argument('--arch', '-a', metavar='ARCH', default='DeepMask', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: DeepMask)')
parser.add_argument('--resume', default='exps/deepmask/train/model_best.pth.tar', help='model to load')


def range_end(start, stop, step=1):
    return np.arange(start, stop+step, step)


def main():
    global args
    args = parser.parse_args()
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup Model
    from collections import namedtuple
    Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch'])
    config = Config(iSz=160, oSz=56, gSz=112, batch=1)  # default for training

    model = (models.__dict__[args.arch](config))
    model = load_pretrain(model, args.resume)
    model = model.eval().to(device)

    scales = [2**i for i in range_end(args.smin, args.smax, args.sstep)]
    meanstd = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    infer = Infer(nps=args.nps, scales=scales, meanstd=meanstd, model=model, device=device, timer=False)

    annFile = '{}/annotations/instances_{}2017.json'.format(args.datadir, args.split)
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    imgIds = sorted(imgIds)  # [:500] for fast test
    segm_props = []
    print('| start eval'); tic = time.time()
    for k, imgId in enumerate(imgIds):
        ann = coco.loadImgs(imgId)[0]
        fileName = ann['file_name']
        pathImg = '{}/{}2017/{}'.format(args.datadir, args.split, fileName)
        im = np.array(Image.open(pathImg).convert('RGB'), dtype=np.float32)
        h, w = im.shape[:2]
        img = np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0).astype(np.float32)
        img = torch.from_numpy(img / 255.).to(device)

        infer.forward(img)
        masks, scores = infer.getTopProps(args.thr, h, w)

        enc = encode(np.asfortranarray(masks))
        for i in range(args.nps):
            enc[i]['counts'] = enc[i]['counts'].decode('utf-8')
            elem = {'segmentation': enc[i], 'image_id': imgId, 'category_id': 1, 'score': scores[i]}
            segm_props.append(elem)

        # for i in range(args.nps):
        #     res = im[:, :, ::-1].copy().astype(np.uint8)
        #     res[:, :, 2] = masks[:, :, i] * 255 + (1 - masks[:, :, i]) * res[:, :, 2]
        #
        #     mask = masks[i].astype(np.uint8)
        #     _, contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #     polygons = [c.reshape(-1, 2) for c in contour]
        #
        #     predict_box = cv2.boundingRect(polygons[0])
        #     predict_rbox = cv2.minAreaRect(polygons[0])
        #     box = cv2.boxPoints(predict_rbox)
        #     print('Segment Proposal Score: {:.3f}'.format(scores[i]))
        #
        #     res = cv2.rectangle(res, (predict_box[0], predict_box[1]),
        #                         (predict_box[0] + predict_box[2], predict_box[1] + predict_box[3]), (0, 255, 0), 3)
        #     res = cv2.polylines(res, [np.int0(box)], True, (0, 255, 255), 3)
        #     cv2.imshow('Proposal', res)
        #     cv2.waitKey(0)
        # plt.imshow(im); plt.axis('off')
        # coco.showAnns([props[-1]])
        # plt.show()
        if (k+1) % 10 == 0:
            toc = time.time() - tic
            print('| process %05d in %010.3f s' % (k+1, toc))

    toc = time.time() - tic
    print('| finish in %010.3f s' % toc)

    pathsv = 'sharpmask/eval_coco' if args.arch == 'SharpMask' else 'deepmask/eval_coco'
    args.rundir = join(args.rundir, pathsv)
    try:
        if not isdir(args.rundir):
            makedirs(args.rundir)
    except OSError as err:
        print(err)

    result_path = join(args.rundir, 'segm_proposals.json')
    with open(result_path, 'w') as outfile:
        json.dump(segm_props, outfile)

    cocoDt = coco.loadRes(result_path)

    print('\n\nBox Proposals Evalution\n\n')
    annType = ['bbox']  # segm  bbox
    cocoEval = COCOeval(coco, cocoDt)

    max_dets = [10, 100, 1000]
    useSegm = False
    useCats = False

    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = max_dets
    cocoEval.params.useSegm = useSegm
    cocoEval.params.useCats = useCats
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print('\n\nSegmentation Proposals Evalution\n\n')
    annType = ['segm']  # segm  bbox
    cocoEval = COCOeval(coco, cocoDt)

    max_dets = [10, 100, 1000]
    useSegm = True
    useCats = False

    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = max_dets
    cocoEval.params.useSegm = useSegm
    cocoEval.params.useCats = useCats
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()
