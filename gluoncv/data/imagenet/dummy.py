"""Dummy classification dataset."""
from os import path
import mxnet as mx
from mxnet.gluon.data.vision import ImageFolderDataset
 
__all__ = ['Dummy']
 
class Dummy(ImageFolderDataset):
    """Load the Dummy classification dataset.
 
    Refer to :doc:`../build/examples_datasets/imagenet` for the description of
    this dataset and how to prepare it.
 
    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/imagenet'
        Path to the folder stored the dataset.
    train : bool, default True
        Whether to load the training or validation set.
    transform : function, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples. (TODO, should we restrict its datatype
        to transformer?)
    """
    def __init__(self, root=path.join('~', '.mxnet', 'datasets', 'imagenet'),
                 train=True, transform=None):
        split = 'train' if train else 'val'
        root = path.join(root, split)
        super(Dummy, self).__init__(root=root, flag=1, transform=transform)
 
    def _list_images(self, root):
        self.items = []
 
        for i in range(1000*1000):
            self.items.append((None, i % 1000))
 
    def __getitem__(self, idx):
        img = mx.nd.zeros((224, 224, 3))
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label
