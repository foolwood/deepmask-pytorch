export PYTHONPATH=.:$PYTHONPATH
python tools/train.py --dataset coco -j 20 --arch SharpMask --freeze_bn
