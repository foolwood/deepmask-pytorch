from tracker.net import SiamRPNvot
from tracker.run_SiamRPN import SiamRPN_init, SiamRPN_track

import glob
import time
import numpy as np
import cv2
import torch
from models import DeepMask
from collections import namedtuple
from utils.load_helper import load_pretrain

import matplotlib.pyplot as plt  # visualization
import matplotlib.patches as patches  # visualization

VISUALIZATION=True


def crop_back(image, bbox, out_sz, padding=0):
    a = (out_sz[0]-1) / bbox[2]
    b = (out_sz[1]-1) / bbox[3]
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=padding)
    return crop


def crop_chw(image, bbox, out_sz, padding=0):
    a = (out_sz-1) / bbox[2]
    b = (out_sz-1) / bbox[3]
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=padding)
    return np.transpose(crop, (2, 0, 1))


def vos(tracker_net, mask_net, image_files, box):

    # tracker init
    target_pos = np.array([box[0]+box[2]/2, box[1]+box[3]/2])
    target_sz = np.array([box[2], box[3]])
    im = cv2.imread(image_files[0])  # HxWxC
    state = SiamRPN_init(im, target_pos, target_sz, tracker_net)

    out_size = 160
    crop_size = 192
    context_amount = 224 / 128
    mean = np.array([[[0.485]], [[0.456]], [[0.406]]]).astype(np.float32)
    std = np.array([[[0.229]], [[0.224]], [[0.225]]]).astype(np.float32)

    toc = 0
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        tic = cv2.getTickCount()
        state = SiamRPN_track(state, im)  # track

        # STAGE2: DeepMask Segmentation
        im_ = (im[:, :, ::-1]).astype(np.float32) / 255.

        target_pos, target_sz = state['target_pos'], state['target_sz']

        max_sz = max(target_sz)
        s_z = context_amount * max_sz
        s_z_c = s_z / out_size * crop_size
        box = [target_pos[0] - s_z / 2, target_pos[1] - s_z / 2, s_z, s_z]
        box_crop = [target_pos[0] - s_z_c / 2, target_pos[1] - s_z_c / 2, s_z_c, s_z_c]
        x_crop = crop_chw(im_, box_crop, crop_size, (0.5, 0.5, 0.5))

        x_crop_norm = (x_crop - mean) / std
        xs_torch = torch.from_numpy(np.expand_dims(x_crop_norm, axis=0))

        mask, score = mask_net(xs_torch.cuda())
        mask = mask.sigmoid().squeeze().cpu().data.numpy()
        # score = score.sigmoid().squeeze().cpu().data.numpy()

        im_w, im_h = im.shape[1], im.shape[0]
        s = out_size / box[2]
        back_box = [-box[0] * s, -box[1] * s, im_w * s, im_h * s]
        mask_in_img = crop_back(mask, back_box, (im_w, im_h))
        toc += cv2.getTickCount() - tic

        if VISUALIZATION:
            plt.cla()
            im_save = im
            plt.imshow(im_save[:, :, ::-1])

            mask = (mask_in_img > 0.3).astype(np.uint8)  # threshold!
            _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                edge = patches.Polygon(contour.reshape(-1, 2), linewidth=2, edgecolor='lawngreen',
                                       facecolor='none')
                ax.add_patch(edge)
                edge = patches.Polygon(contour.reshape(-1, 2), linewidth=2, facecolor='lawngreen',
                                       alpha=0.5)
                ax.add_patch(edge)

            plt.axis('off')
            plt.subplots_adjust(.0, .0, 1, 1)
            plt.draw()
            plt.pause(0.01)

        # if f % 3 == 1:
        #     plt.savefig('%05d.jpg' % f)

    return toc/cv2.getTickFrequency()


if __name__ == "__main__":
    Config = namedtuple('Config', ['iSz', 'oSz', 'gSz'])
    default_config = Config(iSz=160, oSz=56, gSz=160)
    mask_net = DeepMask(default_config)
    mask_net = load_pretrain(mask_net, './pretrained/deepmask/DeepMask.pth.tar')
    mask_net = mask_net.eval().cuda()

    # load net
    tracker_net = SiamRPNvot()
    tracker_net.load_state_dict(torch.load('SiamRPNVOT.model'))
    tracker_net.eval().cuda()

    image_files = sorted(glob.glob('./tracker/bag/*.jpg'))
    tic = time.time()
    if VISUALIZATION:
        try:
            fig
        except NameError:
            fig, ax = plt.subplots(1)

    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', cv2.imread(image_files[0]), False, False)
        x, y, w, h = init_rect
        if not (x | y | w | h): exit()
    except:
        exit()

    toc = vos(tracker_net, mask_net, image_files, init_rect)
    print('Speed: {:.1f} FPS and {:.1f}s'.format(len(image_files)/toc, toc))
    

