import warnings
import torch
from torchvision import transforms as trsf

from .cnn import cnn_toolbox


MEAN_NORMALIZE = [0.485, 0.456, 0.406]
STD_NORMALIZE = [0.229, 0.224, 0.225]

RESIZE_BEFORE_CROP = {224: 256, 32: 36}

################################################################################
# Main
################################################################################

def get_transform(
        mode, size, initial_transform=None, normalize=True, data_augmentation=None,
        resize_before_crop=None, max_shift=None,
        pixel_divide=1, shift_direction=None, downsampling_factor=None,
        return_ref=None, infoprinter=print
):
    """
    Get succession of transforms before feeding images to CNN.

    Parameters
    ----------
    mode (str)
        One of 'train', 'onecrop', 'tencrops', 'shifts', or 'perturbations'.
    size (sequence or int)
        Image size after transform.
    initial_transform (callable, default=None)
    normalize (bool, default=True)
        Whether to normalize tensor image with mean and standard deviation,
        using the same parameters as torchvision's pretrained models. See
        https://pytorch.org/vision/stable/models.html for more details.
    data_augmentation (bool, default=None)
        If True, then randomly crop and flip training data.
        Used when mode is set to 'train', in which case a value must be provided.
    resize_before_crop (int, default=256)
        Used when mode is set to 'tencrops'.
    max_shift (int, default=None)
        Maximum number of pixels in each direction, for image shift. Must be
        provided if 'mode' is set to 'shifts'.
    pixel_divide (int, default=1)
        Number of intermediate shifts per pixel. For example, if set to 2, then
        images will be shifted by steps of 0.5 pixels. This is made possible by
        using bilinear interpolation and antialiased downsampling. See
        https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html
        for more details.
    shift_direction (str, default=None)
        One of 'vertical', 'horizontal' or 'diagonal'. If none is given, then shifts
        will be computed along both directions, which may lead to high memory
        consumption.
    downsampling_factor (int, default=None)
        Downsampling factor of the considered network.
    return_ref (bool, default=None)
        Whether to return the tensor from which patches have been extracted,
        in addition to the patches themselve. Only valid if 'mode' is set to
        'shifts'.
    infoprinter (callable, default=print)

    Returns
    -------
    An instance of torchvision.transforms.Compose

    """
    if resize_before_crop is None:
        try:
            resize_before_crop = RESIZE_BEFORE_CROP[size]
        except KeyError as exc:
            raise ValueError(
                "Argument 'resize_before_crop' must be specified."
            ) from exc
    transf_list = []
    if initial_transform is not None:
        transf_list.append(initial_transform)

    if mode == 'train':
        if data_augmentation is None:
            raise ValueError("Argument 'data_augmentation' not specified")
        if data_augmentation:
            # Randomly crop and flip training data
            transf_list += [
                trsf.RandomResizedCrop(size), # Crop the given PIL Image to random
                                              # size and aspect ratio
                trsf.RandomHorizontalFlip(), # Random flip with probability 0.5
            ]
            transf_list += _totensor(normalize)
        else:
            # Resize and extract a center patch
            transf_list += resize_centercrop(size)
            transf_list += _totensor(normalize)

    elif mode in ('onecrop', 'perturbations'):
        # Resize and extract a center patch
        transf_list += resize_centercrop(size)
        transf_list += _totensor(normalize)

    elif mode == 'tencrops':
        # Same procedure as Krizhevsky et al. (2012): "the network makes a
        # prediction by extracting five 224 × 224 patches (the four corner
        # patches and the center patch) as well as their horizontal
        # reflections (hence ten patches in all), and averaging the
        # predictions made by the network’s softmax layer on the ten patches.
        transf_list += [
            trsf.Resize(resize_before_crop),
            trsf.FiveCrop(size), # this is a list of PIL Images
            ToTensorStackFlip(normalize) # returns a 4D tensor
        ]

    elif mode == 'shifts':
        # Extracts shifted images, in order to evaluate the shift invariance.
        supersize = pixel_divide * size
        assert isinstance(max_shift, int)
        if downsampling_factor is not None:
            max_shift0 = (
                -(-max_shift // downsampling_factor)
            ) * downsampling_factor # max_shift0 is a multiple of downsampling_factor
        else:
            max_shift0 = max_shift
        supermax_shift = pixel_divide * max_shift
        supermax_shift0 = pixel_divide * max_shift0
        transf_list += resize_centercrop(
            size=(supersize + supermax_shift0), resize_before_crop=None
        ) # Get a bigger size than necessary
        transf_list.append(ExtractPatches(
            size=supersize, max_shift=supermax_shift, direction=shift_direction,
            return_ref=return_ref
        ))

        # Antialiased downsampling
        if return_ref:
            size0 = size + max_shift0
        else:
            size0 = None
        transf_list.append(ResizePatchesAndRef(size, size0))

        # Convert to tensor
        transf_list.append(ToTensorPatchesAndRef(return_ref=return_ref))

    else:
        raise ValueError("Unknown evaluation mode.")

    out = trsf.Compose(transf_list)

    infoprinter(
        "\nTransform applied on input data:\n{}\n".format(out), always=True
    )

    return out


def resize_centercrop(size, resize_before_crop=None):
    if resize_before_crop is None:
        resize_before_crop = size
    out = [
        trsf.Resize(resize_before_crop),
        trsf.CenterCrop(size)
    ]
    return out


def _totensor(normalize, apply_on_lists=False):
    """
    Get list of transforms:
    1. Convert to PyTorch tensor;
    2. Normalize tensor if required (only for RGB images).

    """
    out = []
    if not apply_on_lists:
        out.append(trsf.ToTensor())
    else:
        out.append(ToTensorList())
    if normalize:
        out.append(Normalize())
    return out


################################################################################
# Custom transforms
################################################################################

class Normalize(trsf.Normalize):
    def __init__(self, mean=None, std=None, **kwargs):
        if mean is None:
            mean = MEAN_NORMALIZE
        if std is None:
            std = STD_NORMALIZE
        super().__init__(mean, std, **kwargs)


class ToTensorStackFlip:
    """
    Considering a list of PIL images, convert them into PyTorch tensors;
    normalize them if required; stack them and their mirror version along a new
    dimension.

    """
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, crops):

        croplist = []
        for crop in crops:
            transf_list = _totensor(self.normalize)
            crop = trsf.Compose(transf_list)(crop)
            croplist.append(crop)
            croplist.append(torch.flip(crop, dims=(-1,)))

        return torch.stack(croplist)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'normalize={0})'.format(self.normalize)
        return format_string


class ExtractPatches:

    def __init__(
            self, size, max_shift, direction=None, return_ref=False, **kwargs
    ):
        self.size = size
        self.max_shift = max_shift
        self.direction = direction
        self.return_ref = return_ref
        self.kwargs = kwargs

    def __call__(self, inp):
        return cnn_toolbox.extract_patches(
            inp, self.size, self.max_shift + 1, direction=self.direction,
            return_ref=self.return_ref, **self.kwargs
        ) # Nb of extracted images = self.max_shift + 1

    def __repr__(self):
        str_args = 'size={}, max_shift={}'.format(self.size, self.max_shift)
        if self.direction is not None:
            str_args += ', direction={}'.format(self.direction)
        for key, val in self.kwargs.items():
            str_args += ', {}={}'.format(key, val)
        return self.__class__.__name__ + '({})'.format(str_args)


class TransformOnListMixin:
    def __call__(self, inp):
        out = []
        for x in inp:
            out.append(super().__call__(x))
        return out

class ResizeList(TransformOnListMixin, trsf.Resize):
    """Resize the list of input image to the given size."""

class ToTensorList(TransformOnListMixin, trsf.ToTensor):
    """
    Convert a list PIL Image or numpy.ndarray to tensors, and stack them.

    """
    def __call__(self, inp):
        out = super().__call__(inp)
        return torch.stack(out)


class ResizePatchesAndRef:
    """
    Input: (list_of_inp, inp_ref), where 'list_of_inp' is a list of crops
    from 'inp_ref'.

    """
    def __init__(self, size, size0=None):
        self.size = size
        self.size0 = size0
        self.resize_crops = ResizeList(size)
        if size0 is not None:
            self.resize_ref = trsf.Resize(size0)
        else:
            self.resize_ref = None

    def __call__(self, inp):
        if self.size0 is not None:
            inp, inp_ref = inp
        out = self.resize_crops(inp)
        if self.size0 is not None:
            out_ref = self.resize_ref(inp_ref)
            out = out, out_ref
        return out

    def __repr__(self):
        out = self.__class__.__name__ + \
                '(size={}, size_ref={})'.format(self.size, self.size0)
        return out


class ToTensorPatchesAndRef:
    """
    Input: (list_of_inp, inp_ref), where 'list_of_inp' is a list of crops
    from 'inp_ref' (previously downsampled).

    """
    def __init__(self, return_ref):
        self.totensor = ToTensorList()
        if return_ref:
            self.totensor_ref = trsf.ToTensor()
        else:
            self.totensor_ref = None


    def __call__(self, inp):
        if self.totensor_ref is not None:
            inp, inp_ref = inp
        out = self.totensor(inp)
        if self.totensor_ref is not None:
            out_ref = self.totensor_ref(inp_ref)
            out = out, out_ref
        return out


    def __repr__(self):
        return self.__class__.__name__ + '()'
