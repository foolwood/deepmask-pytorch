from .coco_loader import cocoDataset

__dataset = {'coco': cocoDataset}


def get_loader(name):
    return __dataset[name]


def dataset_names():
    return list(__dataset.keys())