import os
import numpy as np
from torchvision import datasets, transforms

from .. import CONFIGDICT

from .. import transforms as customtransf

from .data_structure import CustomDatasetMixin
from .video_imagenet import VideoImageNet

################################################################################
# MNIST
################################################################################

class MNIST(CustomDatasetMixin, datasets.MNIST):
    """
    MNIST dataset.

    Parameters
    ----------
    root (str, default=None)
        Path of the database. If none is given, a config file 'wcnn_config.yml' must
        be provided.
    train (bool, default=True)
        Whether to load the training set (True) or the validation set (False).
    download (bool, default=False)
        If true, downloads the dataset from the internet and puts it in root directory.
        If the dataset is already downloaded, it is not downloaded again.
    totensor (bool, default=False)
        Whether to convert to PyTorch tensor (True) or to keep the data in the
        PIL.Image format (False).

    """
    def __init__(self, root, train=True, download=False, totensor=False):

        if totensor: # Convert to PyTorch tensor
            transform = transforms.ToTensor()
        else: # No transform
            transform = None

        super().__init__(root=root, train=train, download=download, transform=transform)

    @property
    def img_attrs(self):
        return ['data', 'targets']


################################################################################
# CIFAR
################################################################################

N_VAL_CIFAR10 = 5000 # Number of validation images

class CIFAR10(CustomDatasetMixin, datasets.CIFAR10):
    """
    CIFAR10 dataset.

    Arguments
    ---------
    root (str, default=None)
        Path of the database. If none is given, a config file 'wcnn_config.yml' must
        be provided.
    split (str, default='train')
        One of 'train', 'val' or 'test'. The validation set is actually made of the 5000
        first images of the training set provided in
        https://www.cs.toronto.edu/~kriz/cifar.html while the training set is
        made of the 45000 next images.
    rgb (bool, default=True)
        Whether to convert to grayscale (False) or to keep the original RGB format (True).
    download (bool, default=False)
        If true, downloads the dataset from the internet and puts it in root directory. If
        the dataset is already downloaded, it is not downloaded again.

    """
    def __init__(
            self, root=None, split='train', rgb=True, download=False
    ):
        if root is None:
            if CONFIGDICT is not None:
                root = CONFIGDICT["datasets"]["CIFAR10"]
            if root is None:
                raise ValueError("Missing value for argument 'root'.")
        root = os.path.expanduser(root)

        if rgb:
            transform = None
        else:
            transform = transforms.Grayscale()

        if split == 'train':
            train = True
        elif split == 'val':
            train = True
        elif split == 'test':
            train = False

        self._split = split

        super().__init__(root=root, train=train, transform=transform, download=download)

        if split == 'train':
            self.truncate(N_VAL_CIFAR10, keep_until=False, update_classes=False)
        elif split == 'val':
            self.truncate(N_VAL_CIFAR10, keep_until=True, update_classes=False)

    @property
    def img_attrs(self):
        return ['data', 'targets']


################################################################################
# ImageNet ILSVRC2012
################################################################################

# Dictionaries indicating the number of images up to which the dataset must be
# truncated to match a desired number of classes, in the training, validation
# and test set, respectively.
IMAGENET_TRUNC = {
    20: dict(train=24000, val=2000, test=1000),
    100: dict(train=119395, val=10000, test=5000)
}

class _ImageNetMixin(CustomDatasetMixin):
    """
    ImageNet ILSVRC2012 dataset.

    Arguments
    ---------
    root (string)
        Root directory of the ImageNet Dataset.
    split (string, default='train')
        One of 'train', 'val' or 'test'. WARNING: 'val' refers to the validation
        set as specifically defined in this implementation, i.e. a subset of
        images isolated from the training set. 'test' refers to the validation
        set provided by ILSVRC2012.
    n_imgs_per_class (int, default=100)
        Number of images per class in the validation set.
    convert_to_tensor (bool, default=False)
        USE ONLY FOR VISUALIZATION, NOT TRAINING.
        If True, then the loaded data will automatically be cropped converted
        into a PyTorch tensor of size 224 x 224.
    data_folder (str, default=None)
        Path of the folder containing the images, if different from the default
        location.

    """
    def __init__(
        self, root=None, split='train', n_imgs_per_class=100, grayscale=False,
        convert_to_tensor=False, data_folder=None, **kwargs
):
        if root is None:
            if CONFIGDICT is not None:
                root = CONFIGDICT["datasets"]["ImageNet"]["root"]
            if root is None:
                raise ValueError("Missing value for argument 'root'.")
        root = os.path.expanduser(root)

        if split in ('train', 'val'):
            split_orig = 'train'
        elif split == 'test':
            split_orig = 'val'
        else:
            raise ValueError("Unknown split.")

        self._split = split
        self._data_folder = data_folder

        transf_list = []
        if grayscale:
            transf_list.append(transforms.Grayscale())
        if convert_to_tensor:
            transf_list += customtransf.resize_centercrop(size=224)
            transf_list.append(transforms.ToTensor())

        if len(transf_list) > 1:
            transform = transforms.Compose(transf_list)
        elif len(transf_list) == 1:
            transform = transf_list[0]
        else:
            transform = None

        super().__init__(
            root=root, split=split_orig, transform=transform, **kwargs
        )

        # The original training set is split between our custom training and
        # validation sets. More precisely, the 100 first images of each class
        # are set aside for the validation set whereas all the other images
        # remain in the training set.
        if split in ('train', 'val'):
            arr = np.concatenate([[-1], np.array(self.targets)])
            first_idx = np.where(arr[1:] - arr[:-1] != 0)[0]
            for attr in self.img_attrs:
                attr_val = getattr(self, attr)
                new_attr_val = []
                for target, i in enumerate(first_idx):
                    if split == 'train':
                        try:
                            j = first_idx[target + 1]
                        except IndexError:
                            j = len(first_idx)
                        new_attr_val += attr_val[(i + n_imgs_per_class):j]
                    else:
                        new_attr_val += attr_val[i:(i + n_imgs_per_class)]
                setattr(self, attr, new_attr_val)


    @property
    def split_folder(self) -> str:
        if self._data_folder is not None:
            out = self._data_folder
        else:
            out = super().split_folder
        return out

    @property
    def img_attrs(self):
        return ['imgs', 'samples', 'targets']

    @property
    def class_attrs(self):
        return ['classes', 'wnids']

    def _update_classes(self, keep_until):
        # Here we assume that the images in the dataset are sorted by ascending
        # order of class ID.
        class_id = self.targets[-1] + 1 if keep_until else self.targets[0]
        for attr_name in self.class_attrs:
            attr = getattr(self, attr_name)
            attr = attr[:class_id] if keep_until else attr[class_id:]
            setattr(self, attr_name, attr)


class ImageNet(_ImageNetMixin, datasets.ImageNet):
    pass

class _ImageNetFewClasses(ImageNet):
    def __init__(self, num_classes, split='train', **kwargs):
        super().__init__(split=split, **kwargs)
        n_imgs = IMAGENET_TRUNC[num_classes].get(split)
        self.truncate(n_imgs)


class ImageNet100(_ImageNetFewClasses):
    """
    ImageNet dataset with only 100 classes (actually, the 100 first classes of
    the full ImageNet dataset).

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_classes=100, **kwargs)


class ImageNet20(_ImageNetFewClasses):
    """
    ImageNet dataset with only 100 classes (actually, the 20 first classes of
    the full ImageNet dataset).

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_classes=20, **kwargs)


class _ImageNetCPMixin:
    """
    See https://github.com/hendrycks/robustness.

    """
    def __init__(
            self, split, root_cp=None, **kwargs
    ):
        if root_cp is None:
            if CONFIGDICT is not None:
                root_cp = CONFIGDICT["datasets"]["ImageNet"][self.__class__.__name__]
            if root_cp is None:
                raise ValueError("Missing value for argument 'root_cp'.")
        root_cp = os.path.expanduser(root_cp)
        data_folder = os.path.join(root_cp, split)
        super().__init__(split='test', data_folder=data_folder, **kwargs)


    def extra_repr(self) -> str:
        out = "Root location for modified images: {}".format(
            self.split_folder
        )
        return out


class ImageNetC(_ImageNetCPMixin, ImageNet):
    pass

class ImageNetP(_ImageNetCPMixin, _ImageNetMixin, VideoImageNet):
    @property
    def img_attrs(self):
        return ['vids', 'samples', 'targets']
