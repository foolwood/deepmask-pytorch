import os
import time
from os import makedirs
from os.path import isdir, join
import argparse
import logging
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import models
from loader import get_loader, dataset_names
from utils.log_helper import init_log, print_speed, add_file_handler
import matplotlib.pyplot as plt  # visualization
import torchvision.utils as vutils  # visualization
import colorama
from tensorboardX import SummaryWriter

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__") and
                     callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch DeepMask/SharpMask Training')
parser.add_argument('--rundir', default='./exps/', help='experiments directory')
parser.add_argument('--dataset', default='coco', choices=dataset_names(),
                    help='data set')
parser.add_argument('--seed', default=1, type=int, help='manually set RNG seed')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--maxload', default=4000, type=int, metavar='N',
                    help='max number of training batches per epoch')
parser.add_argument('--testmaxload', default=500, type=int, metavar='N',
                    help='max number of testing batches')
parser.add_argument('--maxepoch', default=300, type=int, metavar='N',
                    help='max number of training epochs')
parser.add_argument('--iSz', default=160, type=int, metavar='N',
                    help='input size')
parser.add_argument('--oSz', default=56, type=int, metavar='N',
                    help='output size')
parser.add_argument('--gSz', default=112, type=int, metavar='N',
                    help='ground truth size')
parser.add_argument('--shift', default=16, type=int, metavar='N',
                    help='shift jitter allowed')
parser.add_argument('--scale', default=.25, type=float,
                    help='scale jitter allowed')
parser.add_argument('--hfreq', default=.5, type=float,
                    help='mask/score head sampling frequency')
parser.add_argument('--scratch', action='store_true',
                    help='train DeepMask with randomly initialize weights')
parser.add_argument('--arch', '-a', metavar='ARCH', default='DeepMask',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: DeepMask)')
parser.add_argument('--km', default=32, type=int, help='km')
parser.add_argument('--ks', default=32, type=int, help='ks')
parser.add_argument('--freeze_bn', action='store_true',
                    help='freeze running statistics in BatchNorm layers during training (default: False)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('-v', '--visualize', action='store_true',
                    help='visualize the result heatmap')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BinaryMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = 0
        self.n = 0

    def add(self, output, target):
        target, output = target.squeeze(), output.squeeze()
        assert output.numel() == target.numel(), 'target and output do not match'

        acc = torch.mul(output, target)
        self.acc += acc.ge(0).sum().item()
        self.n += output.size(0)

    def value(self):
        res = self.acc / self.n if self.n > 0 else 0
        return res*100


class IouMeter(object):
    def __init__(self, thr, sz):
        self.sz = sz
        self.iou = torch.zeros(sz, dtype=torch.float32)
        self.thr = np.log(thr / (1 - thr))
        self.reset()

    def reset(self):
        self.iou.zero_()
        self.n = 0

    def add(self, output, target):
        target, output = target.squeeze(), output.squeeze()
        assert output.numel() == target.numel(), 'target and output do not match'

        batch, h, w = output.shape
        pred = output.ge(self.thr)
        mask_sum = pred.eq(1).add(target.eq(1))
        intxn = torch.sum(mask_sum == 2, dim=(1, 2)).float()
        union = torch.sum(mask_sum > 0, dim=(1, 2)).float()
        for i in range(batch):
            if union[i].item() > 0:
                self.iou[self.n+i] = intxn[i].item() / union[i].item()
        self.n += batch

    def value(self, s):
        nb = max(self.iou.ne(0).sum(), 1)
        iou = self.iou.narrow(0, 0, nb)

        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
        if s == 'mean':
            res = iou.mean().item()
        elif s == 'var':
            res = iou.var().item()
        elif s == 'median':
            res = iou.median().squeeze().item()
        elif is_number(s):
            iou_sort, _ = iou.sort()
            res = iou_sort.ge(float(s)).sum().float().item() / float(nb)
        return res * 100


def visual_batch(img, pred_mask, label=None):
    img_show = vutils.make_grid(img, normalize=True, scale_each=True)
    img_show_numpy = np.transpose(img_show.cpu().data.numpy(), axes=(1, 2, 0))

    iSz_res = torch.nn.functional.interpolate(pred_mask, size=(args.iSz, args.iSz))
    pad_res = torch.nn.functional.pad(iSz_res, (16, 16, 16, 16))
    mask_show = vutils.make_grid(pad_res, scale_each=True)
    mask_show_numpy = np.transpose(mask_show.cpu().data.numpy(), axes=(1, 2, 0))

    if str(type(label)).find('torch'):
        iSz_label = torch.nn.functional.interpolate(label, size=(args.iSz, args.iSz))
        pad_label = torch.nn.functional.pad(iSz_label, (16, 16, 16, 16), value=-1)
        label_show = vutils.make_grid(pad_label, normalize=True, scale_each=True)
        label_show_numpy = np.transpose(label_show.cpu().data.numpy(), axes=(1, 2, 0))

        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.imshow(img_show_numpy)
        ax1.imshow(mask_show_numpy[:, :, 0], alpha=.5, cmap='jet')
        # ax1.imshow(mask_show_numpy[:, :, 0] > 0.5, alpha=.5)
        ax1.axis('off')
        ax2.imshow(img_show_numpy)
        ax2.imshow(label_show_numpy, alpha=.5)
        ax2.axis('off')
    else:
        plt.imshow(img_show_numpy)
        plt.imshow(mask_show_numpy[:, :, 0], alpha=.5, cmap='jet')
        # ax1.imshow(mask_show_numpy[:, :, 0] > 0.2, alpha=.5)
        plt.axis('off')
    plt.subplots_adjust(.05, .05, .95, .95)
    plt.show()
    plt.close()


def BNtoFixed(m):
    # From https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/torchtools.py
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.eval()


def train(train_loader, model, criterion, optimizer, epoch):
    logger = logging.getLogger('global')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mask_losses = AverageMeter()
    score_losses = AverageMeter()

    # switch to train mode
    model.train()
    if args.freeze_bn:
        model.apply(BNtoFixed)
    train_loader.dataset.shuffle()

    end = time.time()
    for i, (img, target, head_status) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img = img.to(device)
        target = target.to(device)

        # compute output
        output = model(img)
        loss = criterion(output[head_status[0]], target)

        # measure and record loss
        if head_status[0] == 0:
            mask_losses.update(loss.item(), img.size(0))
            loss.mul_(img.size(0))
        else:
            score_losses.update(loss.item(), img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()  # gradOutputs:mul(self.inputs:size(1))
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # REMOVE?
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.visualize and head_status[0] == 0:
            visual_batch(img, output[0].sigmoid(), target)

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\n'
                        'LR {lr:.1e} \t Mask Loss {mask_loss.val:.4f} ({mask_loss.avg:.4f})\t'
                        'Score Loss {score_loss.val:.3f} ({score_loss.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, lr=optimizer.param_groups[0]['lr'],
                data_time=data_time, mask_loss=mask_losses, score_loss=score_losses))
            print_speed(epoch * len(train_loader) + i + 1, batch_time.avg, args.maxepoch * len(train_loader))

    writer.add_scalar('train_loss/mask_loss', mask_losses.avg, epoch)
    writer.add_scalar('train_loss/score_losses', score_losses.avg, epoch)


def validate(val_loader, model, criterion, epoch=0):
    logger = logging.getLogger('global')
    batch_time = AverageMeter()
    mask_losses = AverageMeter()
    score_losses = AverageMeter()

    mask_meter = IouMeter(0.5, len(val_loader.dataset))
    score_meter = BinaryMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (img, target, head_status) in enumerate(val_loader):
            img = img.to(device)
            target = target.to(device)

            # compute output
            output = model(img)
            loss = criterion(output[head_status[0]], target)

            # measure accuracy and record loss
            if head_status[0] == 0:
                mask_losses.update(loss.item(), img.size(0))
                mask_meter.add(output[head_status[0]], target)
            else:
                score_losses.update(loss.item(), img.size(0))
                score_meter.add(output[head_status[0]], target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        acc = mask_meter.value('0.7')

        logger.info(' * Epoch [{0}]Mask Loss {mask_loss.avg:.3f} Score Loss {mask_loss.avg:.3f}'.format(
            epoch, mask_loss=mask_losses, cls_loss=score_losses))
        logger.info(' * Epoch [%03d] | IoU: mean %05.2f median %05.2f suc@.5 %05.2f suc@.7 %05.2f '
                    '| acc %05.2f | bestmodel %s' % (epoch, mask_meter.value('mean'),
                    mask_meter.value('median'), mask_meter.value('0.5'), mask_meter.value('0.7'),
                    score_meter.value(), 'y' if acc > max_acc else 'n'))

        writer.add_scalar('val_loss/mask_loss', mask_losses.avg, epoch)
        writer.add_scalar('val_loss/score_losses', score_losses.avg, epoch)
        writer.add_scalar('val_meter/mask_meter_mean', mask_meter.value('mean'), epoch)
        writer.add_scalar('val_meter/mask_meter_median', mask_meter.value('median'), epoch)
        writer.add_scalar('val_meter/mask_meter_0.5', mask_meter.value('0.5'), epoch)
        writer.add_scalar('val_meter/mask_meter_0.7', mask_meter.value('0.7'), epoch)
        writer.add_scalar('val_meter/score_meter', score_meter.value(), epoch)

    return acc


def save_checkpoint(state, is_best, file_path='', filename='checkpoint.pth.tar'):
    torch.save(state, join(file_path, filename))
    if is_best:
        shutil.copyfile(join(file_path, filename), join(file_path, 'model_best.pth.tar'))


def main():
    global args, device, max_acc, writer

    max_acc = -1
    args = parser.parse_args()
    if args.arch == 'SharpMask':
        trainSm = True
        args.hfreq = 1
        args.gSz = args.iSz
    else:
        trainSm = False

    # Setup experiments results path
    pathsv = 'sharpmask/train' if trainSm else 'deepmask/train'
    args.rundir = join(args.rundir, pathsv)
    try:
        if not isdir(args.rundir):
            makedirs(args.rundir)
    except OSError as err:
        print(err)

    # Setup logger
    init_log('global', logging.INFO)
    add_file_handler('global', join(args.rundir, 'train.log'), logging.INFO)
    logger = logging.getLogger('global')
    logger.info('running in directory %s' % args.rundir)
    logger.info(args)
    writer = SummaryWriter(log_dir=join(args.rundir, 'tb'))

    # Get argument defaults (hastag #thisisahack)
    parser.add_argument('--IGNORE', action='store_true')
    defaults = vars(parser.parse_args(['--IGNORE']))

    # Print all arguments, color the non-defaults
    for argument, value in sorted(vars(args).items()):
        reset = colorama.Style.RESET_ALL
        color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
        logger.info('{}{}: {}{}'.format(color, argument, value, reset))

    # Setup seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup Model
    model = (models.__dict__[args.arch](args)).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    logger.info(model)

    # Setup data loader
    train_dataset = get_loader(args.dataset)(args, split='train')
    val_dataset = get_loader(args.dataset)(args, split='val')
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch, num_workers=args.workers,
        pin_memory=True, sampler=None)
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch, num_workers=args.workers,
        pin_memory=True, sampler=None)

    # Setup Metrics
    criterion = nn.SoftMarginLoss().to(device)

    # Setup optimizer, lr_scheduler and loss function
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[50, 120], gamma=0.3)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            max_acc = checkpoint['max_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            logger.warning("no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.maxepoch):
        scheduler.step(epoch=epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if epoch % 2 == 1:
            acc = validate(val_loader, model, criterion, epoch)

            is_best = acc > max_acc
            max_acc = max(acc, max_acc)
            # remember best mean loss and save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'max_acc': max_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.rundir)


if __name__ == '__main__':
    main()
