"""
Objects replacing standard layers in CNNs. They inherit from torch.nn.Module.

Classes
-------
WptBlock (also inherits from .base_modules.ConvKernelMixin)
    Wavelet packet transform followed by a mixing layer, replacing the first
    freely-trained convolution layer.
Pooling
    Non-linear module replacing ReLU + MaxPool2d.

"""
from collections import OrderedDict
import torch
from torch import nn

from .. import classifier, toolbox
from . import cnn_toolbox, base_modules, wpt

WPT_TYPE = {
    'wpt2d': wpt.Wpt2d,
    'dtcwpt2d': wpt.DtCwpt2d
}

# List of keyword arguments which are specific to the WPT and mixing modules,
# respectively
LIST_OF_KWARGS_WPT = ['padding_mode']
LIST_OF_KWARGS_FEATMAPMIX = [
    'groups', 'bias', 'split_out_featmaps', 'out_featmaps'
]

class WptBlock(base_modules.Sequential):
    """
    A wavelet packet transform module.

    Parameters
    ----------
    in_channels (int)
        Number of input channels (3 for RGB images).

    wpt_type (str)
        Type of wavelet packet transform: 'wpt2d' or 'dtcwpt2d'

    depth (int)
        Number of decomposition levels in the wavelet packet transform

    stride (int)
        Subsampling factor

    permute_featmaps (list, default=None)
        Indicates how to permute wavelet packet feature maps before performing
        filter selection. The list has as many elements as the number of kept
        feature maps. If none is given, then no permutation is done.

    split_in_featmaps (list of int, default=None)
        Only used if wpt_type == 'dtcwpt2d'. Provides the number of wavelet
        packet feature maps for each group.

    split_out_featmaps (list of int, default=None)
        Only used if wpt_type == 'dtcwpt2d'. Provides the number of output
        feature maps for each group. The number of output channels is deduced
        from this parameter as well as 'wpt_type', 'complex_featmaps' and
        'real_part_only' parameters (see below). More precisely,
            - if wpt_type == 'dtcwpt2d', complex_featmaps is True and real_part_only
              is also True, then the output of this layer is made of complex-valued
              feature maps; each of which being represented as a stack of two real-
              valued output channels;
            - in any other cases, the output is made of real-valued feature maps;
              each of which being represented as a single output channel.

    out_featmaps (int, default=None)
        Number of output feature maps. This parameter must be provided if
        split_out_featmaps is None, in which case the output feature maps will
        be split evenly between each group.

    groups (int or list of int, default=None)
        For each bunch of channels, number of blocked connections from input
        channels to output channels. See torch.nn.Conv2d for more details.

    real_part_only (bool, default=False)

    color_mixing_before_wpt (bool, default=False)

    init_colormix_weight (torch.Tensor, default=None)
        Only used when 'color_mixing_before_wpt' is set to True. PyTorch tensor
        of shape (1, in_channels, 1, 1).

    freeze_colormix_weights (bool, default=False)
        Only used when 'color_mixing_before_wpt' is set to True. Whether to
        freeze the color mixing parameters to their initial value.

    wavelet (str, default=None)
        See dtcwt.coeffs.qshift for more details.

    backend (str, default='dtcwt')
        One of 'pywt' or 'dtcwt'. Name of the library from which to get the
        filters.

    trainable_qmf (bool, default=False)
        Whether to set the QMFs as trainable parameters.

    padding_mode (str, default='zeros')
        Padding mode in the WPT layer. One of 'zeros' (default) or 'symmetric'.

    """
    def __init__(self, in_channels, wpt_type, depth, stride,
                 permute_featmaps=None, split_in_featmaps=None,
                 real_part_only=False, color_mixing_before_wpt=False,
                 init_colormix_weight=None, freeze_colormix_weights=False,
                 **kwargs):

        kwargs_wpt = toolbox.extract_dict(LIST_OF_KWARGS_WPT, kwargs)
        kwargs_featmapmix = toolbox.extract_dict(
            LIST_OF_KWARGS_FEATMAPMIX, kwargs
        )

        n_colors = in_channels # Variable in_channel will be updated

        layers = []

        # Color mixing (if required before WPT)
        if color_mixing_before_wpt:
            color_mixlayer = base_modules.Conv2d(
                in_channels, out_channels=1, kernel_size=1, bias=False
            )
            assert color_mixlayer.stride == (1, 1)
            if init_colormix_weight is not None:
                state_dict = color_mixlayer.state_dict()
                state_dict['weight'] = init_colormix_weight
                color_mixlayer.load_state_dict(state_dict)
            if freeze_colormix_weights:
                for param in color_mixlayer.parameters():
                    param.requires_grad = False
            layers.append(('colormix', color_mixlayer))
            in_channels = color_mixlayer.get_out_channels()

        # WPT layer
        wpt_layer = WPT_TYPE[wpt_type](
            in_channels, depth, stride, **kwargs_wpt, **kwargs
        )
        layers.append(('wpt', wpt_layer))
        in_channels = wpt_layer.get_out_channels()

        complex_featmaps = False
        if wpt_type == 'dtcwpt2d':
            complex_featmaps = True
        else:
            complex_featmaps = False

        # Permute feature maps in order to group RGB channels
        if not color_mixing_before_wpt:
            grouprgblayer = base_modules.GroupRGBChannels(
                in_channels, n_colors=n_colors, complex_featmaps=complex_featmaps
            )
            layers.append(('group_rgb', grouprgblayer))

        # Permute wavelet packet feature maps, if required
        if permute_featmaps is not None:
            if not color_mixing_before_wpt:
                in_featmaps = in_channels // n_colors
            else:
                in_featmaps = in_channels
            if complex_featmaps:
                in_featmaps //= 2
            permutelayer = base_modules.PermuteFeatmaps(
                in_channels, in_featmaps, permute_featmaps
            )
            layers.append(('permute_featmaps', permutelayer))
            in_channels = permutelayer.get_out_channels()

        # Feature map mixing module
        featmap_mixlayer = base_modules.MixModule(
            in_channels, split_in_featmaps, **kwargs_featmapmix
        )
        layers.append(('featmapmix', featmap_mixlayer))
        in_channels = featmap_mixlayer.get_out_channels()

        # Color mixing (if required after WPT)
        if not color_mixing_before_wpt:
            if complex_featmaps:
                convtype = base_modules.ComplexConv2d
                in_featmaps = in_channels // 2
            else:
                convtype = base_modules.Conv2d
                in_featmaps = in_channels

            out_featmaps = in_featmaps // n_colors

            color_mixlayer = convtype(
                in_featmaps, out_featmaps, kernel_size=1, groups=out_featmaps,
                bias=False
            )
            assert color_mixlayer.stride == (1, 1)
            layers.append(('colormix', color_mixlayer))
            in_channels = color_mixlayer.get_out_channels()

        # Get split output sections
        self._split_out_sections = featmap_mixlayer.split_out_sections
        if not color_mixing_before_wpt:
            self._split_out_sections = [
                sec // n_colors for sec in self._split_out_sections
            ]

        # Select real part, if required. Argument 'real_pqart_only' can be a
        # list of booleans (in which case the condition becomes true)
        if wpt_type == 'dtcwpt2d' and real_part_only:
            realpartlayer = base_modules.RealPartOnly(
                in_channels=in_channels,
                split_in_sections=self._split_out_sections,
                real_part_only=real_part_only
            )
            layers.append(('realpart', realpartlayer))
            self._split_out_sections = realpartlayer.split_out_sections

        super().__init__(OrderedDict(layers))

    @property
    def split_out_sections(self):
        return self._split_out_sections

    @property
    def weights_l1reg(self):
        return self.featmapmix.weights_l1reg


# List of keyword arguments to be passed to the freely_trained convolution layer
LIST_OF_KWARGS_CONV2D = ['padding']

# List of keyword arguments to be passed to all layers
LIST_OF_KWARGS_CONV2DWPTBLOCK = ['padding_mode', 'bias']

class HybridConv2dWpt(base_modules.ParallelConv2d):
    """
    Convolution-like module in which some output channels are computed with
    freely-trained convolution kernels whereas others are computed with a WPT
    block.

    Parameters
    ----------
    in_channels (int)
        Number of input channels (3 for RGB images).

    out_channels_freeconv (int)
        Number of channels produced by the freely-trained convolution layer.

    kernel_size_freeconv (int or tuple)
        Size of the freely-trained convolving kernel

    wpt_type (str)
        Type of wavelet packet transform: 'wpt2d' or 'dtcwpt2d'

    depth (int)
        Number of decomposition levels in the wavelet packet transform

    stride (int)
        Subsampling factor

    pretrained_model_freeconv (int, default=None)

    status_pretrained_freeconv (str, default=None)
        Must be specified if 'pretrained_model_freeconv' is provided.

    skip_pretrained_freeconv (bool, default=False)
        Set this parameter to True when loading a trained classifier.
        The pretrained parameters would be overwritten by the state
        dictionary anyway.

    list_of_channels_pretrained_freeconv (list of int, default=None)
        Must be specified if 'pretrained_model_freeconv' is provided.
        List of output channels from which the freely-trained kernels will be
        initialized. Its length must be equal to 'out_channels_freeconv'.

    freeze_freeconv (bool, default=True)
        Whether to freeze the freely-trained convolution kernels to their
        initial state.

    padding (int, tuple or stride, default=0)
        Padding added to all four sides of the input (freely-trained layer).

    padding_mode (str, default='zeros')
        Padding mode (all layers). One of 'zeros' (default) or 'symmetric'.

    split_in_featmaps (list of int, default=None)
        Only used if wpt_type == 'dtcwpt2d'. Provides the number of wavelet
        packet feature maps for each group.

    split_out_featmaps (list of int, default=None)
        Only used if wpt_type == 'dtcwpt2d'. Provides the number of output
        feature maps for each group. The number of output channels is deduced
        from this parameter as well as 'wpt_type', 'complex_featmaps' and
        'real_part_only' parameters (see below). More precisely,
            - if wpt_type == 'dtcwpt2d', complex_featmaps is True and real_part_only
              is also True, then the output of this layer is made of complex-valued
              feature maps; each of which being represented as a stack of two real-
              valued output channels;
            - in any other cases, the output is made of real-valued feature maps;
              each of which being represented as a single output channel.

    out_featmaps (int, default=None)
        Number of output feature maps. This parameter must be provided if
        split_out_featmaps is None, in which case the output feature maps will
        be split evenly between each group.

    groups (int or list of int, default=None)
        For each bunch of channels, number of blocked connections from input
        channels to output channels. See torch.nn.Conv2d for more details.

    real_part_only (bool, default=False)

    wavelet (str, default=None)
        See dtcwt.coeffs.qshift for more details.

    backend (str, default='dtcwt')
        One of 'pywt' or 'dtcwt'. Name of the library from which to get the
        filters.

    trainable_qmf (bool, default=False)
        Whether to set the QMFs as trainable parameters.

    padding_mode (str, default='zeros')
        Padding mode in the WPT layer. One of 'zeros' (default) or 'symmetric'.

    """
    def __init__(
            self, in_channels, out_channels_freeconv, kernel_size_freeconv,
            wpt_type, depth, stride, pretrained_model_freeconv=None,
            status_pretrained_freeconv=None, skip_pretrained_freeconv=False,
            list_of_channels_pretrained_freeconv=None, freeze_freeconv=True,
            bias_wptblock=None, **kwargs
    ):
        kwargs_conv2d = toolbox.extract_dict(LIST_OF_KWARGS_CONV2D, kwargs)
        kwargs_conv2dwptblock = toolbox.extract_dict(
            LIST_OF_KWARGS_CONV2DWPTBLOCK, kwargs
        )

        # Freely-trained convolution layer
        #=======================================================================
        freeconv = base_modules.Conv2d(
            in_channels, out_channels_freeconv, kernel_size_freeconv,
            stride=stride, **kwargs_conv2d, **kwargs_conv2dwptblock
        )
        if pretrained_model_freeconv is not None and not skip_pretrained_freeconv:
            kwargs_loadpretrained = toolbox.extract_dict(
                classifier.LIST_OF_KWARGS_LOAD, kwargs
            )
            conv0 = classifier.WaveCnnClassifier.load(
                pretrained_model_freeconv, status=status_pretrained_freeconv,
                warn=False, **kwargs_loadpretrained
            ).net_.first_conv_layer # First conv layer of the pretrained model
            sd_pretrained = conv0.state_dict()

            # Select the desired channels from which to initialize the freely-
            # trained convolution kernels
            assert len(list_of_channels_pretrained_freeconv) \
                    == out_channels_freeconv
            for param in sd_pretrained:
                sd_pretrained[param] = sd_pretrained[param][
                    list_of_channels_pretrained_freeconv
                ]

            # Load state dictionary
            freeconv.load_state_dict(sd_pretrained)

            # Freeze weights if required
            if freeze_freeconv:
                for param in freeconv.parameters():
                    param.requires_grad = False

        # Wavelet block
        #=======================================================================
        if bias_wptblock is not None:
            kwargs.update(bias=bias_wptblock)
        wptblock = WptBlock(
            in_channels, wpt_type, depth, stride,
            **kwargs_conv2dwptblock, **kwargs
        )

        convdict = OrderedDict({"freeconv":freeconv, "wptblock":wptblock})

        super().__init__(in_channels, stride, convdict)


    @property
    def split_out_sections(self):
        return [self.freeconv.out_channels] + self.wptblock.split_out_sections


    @property
    def weights_l1reg(self):
        return self.wptblock.weights_l1reg


class Pooling(nn.ModuleDict):
    """
    Parameters
    ----------
    split_sections (list of int)
    whichtypes (list of str)
        For each group of input channels, indicates whether they correspond to
        low- ('low') or high-frequency ('high') feature maps.
    pooling_low, pooling_high (torch.nn.Module)
        Pooling layers through which low- and high-frequency feature maps
        must be passed, respectively.

    """
    def __init__(self, split_sections, whichtypes, pooling_low,
                 pooling_high):

        assert len(whichtypes) == len(split_sections)
        for which in whichtypes:
            assert which in ('low', 'high')

        split_sections, whichtypes = cnn_toolbox.merge_sections(
            split_sections, whichtypes
        )

        self._split_sections = split_sections
        self._whichtypes = whichtypes

        super().__init__({
            'low': pooling_low,
            'high': pooling_high
        })


    def forward(self, x):

        inp = torch.split(x, self._split_sections, dim=1)
        out = []
        for x0, whichtype in zip(inp, self._whichtypes):
            x0 = x0.clone() # Avoids RuntimeError due to inplace modifications
            out.append(self[whichtype](x0))

        # Crop feature maps before concatenation
        cnn_toolbox.crop_featmaps_before_concat(out)

        out = torch.cat(out, dim=1)

        return out


    def extra_repr(self):
        splitsecs = []
        for n_channels, which in zip(self._split_sections, self._whichtypes):
            splitsecs.append('{} ({})'.format(n_channels, which))
        splitsecs = ', '.join(splitsecs)
        out = 'in_channels: {}'.format(splitsecs)
        return out
