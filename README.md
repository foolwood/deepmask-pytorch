# deepmask-pytorch

This repository contains a PyTorch re-implementation for the [DeepMask](https://arxiv.org/abs/1506.06204) and [SharpMask](https://arxiv.org/abs/1603.08695) object proposal algorithms.

<div align="center">
  <img src="data/heatmap.png" width="700px" />
</div>

## Requirements and Dependencies
* Linux
* NVIDIA GPU with compute capability 3.5+
* python3
* [PyTorch 0.4.1](https://pytorch.org/)

## Quick Start
To run pretrained `DeepMask` models to generate object proposals, follow these steps:

1. Clone this repository into $DEEPMASK:

   ```bash
   git clone https://github.com/foolwood/deepmask-pytorch.git
   cd deepmask-pytorch
   DEEPMASK=$PWD
   export PYTHONPATH=$DEEPMASK:$PYTHONPATH
   ```

2. Download pre-trained DeepMask models:

   ```bash
   mkdir -p $DEEPMASK/pretrained/deepmask; cd $DEEPMASK/pretrained/deepmask
   wget http://www.robots.ox.ac.uk/~qwang/DeepMask.pth.tar
   ```

3. Run `computeProposals.py` with a given model and optional target image (specified via the `-img` option):

   ```bash
   # apply to a default sample image (data/testImage.jpg)
   cd $DEEPMASK
   python tools/computeProposals.py --arch DeepMask --resume $DEEPMASK/pretrained/deepmask/DeepMask.pth.tar --img ./data/test.jpg
   ```

## Training Your Own Model
To train your own `DeepMask` models, follow these steps:

### Preparation
1. If you have not done so already, clone this repository into $DEEPMASK:

   ```bash
   git clone https://github.com/foolwood/deepmask-pytorch.git
   cd deepmask-pytorch
   DEEPMASK=$PWD
   export PYTHONPATH=$DEEPMASK:$PYTHONPATH
   ```

2. Download and extract the [COCO](http://mscoco.org/) images and annotations:

   ```bash
   mkdir -p $DEEPMASK/data/coco; cd $DEEPMASK/data/coco
   wget http://images.cocodataset.org/zips/train2017.zip
   wget http://images.cocodataset.org/zips/val2017.zip
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

   unzip ./train2017.zip && unzip ./val2017.zip && unzip ./annotations_trainval2017.zip
   cd $DEEPMASK/loader/pycocotools && make
   ```

### Training
To train DeepMask, launch the `train.py` script. It contains several options, to list them, simply use the `--help` flag.

```bash
cd $DEEPMASK
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --dataset coco -j 20 --freeze_bn
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --dataset coco -j 20 --arch SharpMask --freeze_bn
```

#### Testing
Test on COCO validation set (5K).
```bash
sh scripts/test_recall_coco.sh
```
Our results
```shell
                +-----------------+------------------------+------------------------+
                |                 |      Box Proposals     | Segmentation Proposals |
                +                 +------------------------+------------------------+
                |                 | AR_10 | AR_100 | AR_1K | AR_10 | AR_100 | AR_1K |
                +-----------------+-------+--------+-------+-------+--------+-------+
                | DeepMask(paper) |  18.7 |  34.9  |  46.5 |  14.4 |  25.8  |  33.1 |
                +-----------------+-------+--------+-------+-------+--------+-------+
                | DeepMask(ours)  |  18.3 |  35.9  |  48.4 |  13.6 |  26.0  |  33.5 |
                +-----------------+-------+--------+-------+-------+--------+-------+
```

## Naive Cascade Instance Segmentation (YOLOv3+DeepMask=10FPS~28FPS)

<div align="center">
  <img src="data/00129.jpg" width="250px" />
  <img src="data/00149.jpg" width="250px" />
  <img src="data/00182.jpg" width="250px" />
  <img src="data/00311.jpg" width="250px" />
  <img src="data/00358.jpg" width="250px" />
  <img src="data/00507.jpg" width="250px" />
</div>

```bash
git clone https://github.com/pjreddie/darknet.git
cd darknet
make # Compile with CUDA https://pjreddie.com/darknet/install/
sed -i 's/= data/= .\/darknet\/data/g' cfg/coco.data
sed -i 's/batch=64/batch=1/g' cfg/yolov3.cfg
sed -i 's/subdivisions=16/subdivisions=1/g' cfg/yolov3.cfg
cd $DEEPMASK
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
python tools/yolo_deepmask.py
```

## Naive Cascade Video Object Segmentation (DaSiamRPN+DeepMask=60FPS)

<div align="center">
  <img src="data/vos.gif" width="400px" />
</div>

```bash
git clone https://github.com/foolwood/DaSiamRPN.git
mkdir tracker && mv DaSiamRPN/code/* ./tracker/ && rm -rf ./DaSiamRPN
touch tracker/__init__.py
sed -i 's/utils/.utils/g' tracker/run_SiamRPN.py
cd $DEEPMASK
wget www.robots.ox.ac.uk/~qwang/SiamRPNVOT.model
python tools/dasiamrpn_deepmask.py
```


## Citations
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.
```
@misc{wang2018deepmask,
author = {Wang Qiang},
title = {{deepmask-pytorch}},
year = {2018},
howpublished = {\url{https://github.com/foolwood/deepmask_pytorch}}
}
```
```
@inproceedings{DeepMask,
   title = {Learning to Segment Object Candidates},
   author = {Pedro O. Pinheiro and Ronan Collobert and Piotr Dollár},
   booktitle = {NIPS},
   year = {2015}
}
@inproceedings{SharpMask,
   title = {Learning to Refine Object Segments},
   author = {Pedro O. Pinheiro and Tsung-Yi Lin and Ronan Collobert and Piotr Dollár},
   booktitle = {ECCV},
   year = {2016}
}
```

## License

Licensed under an MIT license
