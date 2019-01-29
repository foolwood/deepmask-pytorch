from ctypes import *
import random

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

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


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
    return np.transpose(crop, (2, 0, 1))


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, mask_model=None):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])

    # STAGE2: DeepMask Segmentation
    im_raw = cv2.imread(image.decode("utf-8"))
    im_ = (im_raw[:, :, ::-1]).astype(np.float32) / 255.
    out_size = 160
    crop_size = 192
    context_amount = 224 / 128
    mean = np.array([[[0.485]], [[0.456]], [[0.406]]]).astype(np.float32)
    std = np.array([[[0.229]], [[0.224]], [[0.225]]]).astype(np.float32)
    masks_in_img = []
    for i, a in enumerate(res):
        b = a[2]
        cx, cy, w, h = np.array(b)
        x = cx - w / 2
        y = cy - h / 2
        x, y, w, h = int(x), int(y), int(w), int(h)

        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])

        max_sz = max(target_sz)

        s_z = context_amount * max_sz
        s_z_c = s_z / out_size * crop_size
        box = [target_pos[0] - s_z / 2, target_pos[1] - s_z / 2, s_z, s_z]
        box_crop = [target_pos[0] - s_z_c / 2, target_pos[1] - s_z_c / 2, s_z_c, s_z_c]
        x_crop = crop_chw(im_, box_crop, crop_size, (0.5, 0.5, 0.5))

        x_crop_norm = (x_crop - mean) / std
        xs_torch = torch.from_numpy(np.expand_dims(x_crop_norm, axis=0))

        mask, score = model(xs_torch.cuda())
        mask = mask.sigmoid().squeeze().cpu().data.numpy()
        # score = score.sigmoid().squeeze().cpu().data.numpy()

        im_w, im_h = im_raw.shape[1], im_raw.shape[0]
        s = out_size / box[2]
        back_box = [-box[0] * s, -box[1] * s, im_w * s, im_h * s]
        mask_in_img = crop_back(mask, back_box, (im_w, im_h))
        masks_in_img.append(mask_in_img)

    if VISUALIZATION:
        plt.cla()
        im_save = im_raw
        plt.imshow(im_save[:, :, ::-1])

        for id in range(len(masks_in_img)):
            mask = (masks_in_img[id] > 0.3).astype(np.uint8)
            _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                mvos_color = ['r', 'lawngreen', 'aqua', 'fuchsia', 'yellow', 'blue'][id%6]
                edge = patches.Polygon(contour.reshape(-1, 2), linewidth=2, edgecolor=mvos_color,
                                       facecolor='none')
                ax.add_patch(edge)
                edge = patches.Polygon(contour.reshape(-1, 2), linewidth=2, facecolor=mvos_color,
                                       alpha=0.5)
                ax.add_patch(edge)

        for id, a in enumerate(res):
            cx, cy, w, h = np.array(a[2])
            x = cx - w / 2
            y = cy - h / 2
            mvos_color = ['r', 'lawngreen', 'aqua', 'fuchsia', 'yellow', 'blue'][id % 6]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=mvos_color, facecolor='none')
            ax.add_patch(rect)

        plt.axis('off')
        plt.subplots_adjust(.0, .0, 1, 1)
        plt.draw()
        plt.pause(0.5)

    free_image(im)
    free_detections(dets, num)
    return res


if __name__ == "__main__":
    Config = namedtuple('Config', ['iSz', 'oSz', 'gSz'])
    default_config = Config(iSz=160, oSz=56, gSz=160)
    model = DeepMask(default_config)
    model = load_pretrain(model, './pretrained/deepmask/DeepMask.pth.tar')
    model = model.eval().to('cuda')

    net = load_net(b"./darknet/cfg/yolov3-tiny.cfg", b"yolov3-tiny.weights", 0)
    # net = load_net(b"./darknet/cfg/yolov3.cfg", b"yolov3.weights", 0)
    meta = load_meta(b"./darknet/cfg/coco.data")
    image_files = glob.glob('./data/coco/val2017/*.jpg')
    tic = time.time()
    if VISUALIZATION:
        try:
            fig
        except NameError:
            fig, ax = plt.subplots(1)

    for i, image_file in enumerate(image_files):
        r = detect(net, meta, image_file.encode(), mask_model=model)
        print(r)
        # plt.savefig('%05d.jpg' % i)
    toc = time.time() - tic
    print('Speed: {:.1f} FPS and {:.1f}s'.format(len(image_files)/toc, toc))
    

