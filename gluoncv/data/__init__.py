"""
This module provides data loaders and transfomers for popular vision datasets.
"""
from . import transforms
from .imagenet.classification import ImageNet
from .imagenet.dummy import Dummy
from .dataloader import DetectionDataLoader
from .pascal_voc.detection import VOCDetection
from .mscoco.detection import COCODetection
from .pascal_voc.segmentation import VOCSegmentation
from .pascal_aug.segmentation import VOCAugSegmentation
from .ade20k.segmentation import ADE20KSegmentation
from .segbase import get_segmentation_dataset, test_batchify_fn
