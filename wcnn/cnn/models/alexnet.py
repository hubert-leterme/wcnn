from typing import OrderedDict
import torch
from torch import nn
from torchvision import models
import antialiased_cnns as zhangmodels

from .. import cnn_toolbox, base_modules, building_blocks

from . import model_toolbox

#===============================================================================
# Base classes
#===============================================================================

class _BaseAlexNetMixin(model_toolbox.CustomModelMixin):

    @property
    def first_conv_layer(self):
        return self.features[0]

    def _get_list_of_kwargs_net(self):
        out = [
            'blurfilt_size', 'antialiased_firstconv'
        ] # Only for ZhangAlexNet
        return out

    def _keep_kwargs_for_structure_modifications(self):
        return ['blurfilt_size']

    def _replace_last_layer(self, num_classes):
        old_fc = self.classifier[6]
        self.classifier[6] = self._new_last_layer(old_fc, num_classes)


class _BaseZhangAlexNet(zhangmodels.AlexNet):

    def __init__(
            self, blurfilt_size=3, antialiased_firstconv=True,
            **kwargs
    ):
        if blurfilt_size is not None:
            kwargs.update(filter_size=blurfilt_size)
        super().__init__(**kwargs)

        if not antialiased_firstconv:
            # Stride = 4 instead of 2 in the first convolution layer
            self.features[0] = nn.Conv2d(
                3, 64, kernel_size=11, stride=4, padding=2
            )
            # Remove first blurpool layer
            module_list = list(self.features.children())
            module_list.pop(2)
            self.features = nn.Sequential(*module_list)


#===============================================================================
# Standard AlexNet
#===============================================================================

# List of arguments for the first convolution layer
ARGS_CONV0 = [
    'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
    'padding_mode', 'dilation', 'groups'
]

CONFIG_DICT_STDALEXNET = {
    'Std': {},
    'Bf2': dict(blurfilt_size=2),
    'Bf3': dict(blurfilt_size=3),
    'Bf4': dict(blurfilt_size=4),
    'Bf5': dict(blurfilt_size=5),
    'Bf6': dict(blurfilt_size=6),
    'Bf7': dict(blurfilt_size=7),
    'Str4': dict(antialiased_firstconv=False),
    'Ks20': dict(kernel_size=20),
    'Frozen': dict(n_frozen_layers=1)
}

class _BaseStdAlexNetMixin(_BaseAlexNetMixin):

    def __init__(self, config='Std', **kwargs):
        if config is not None:
            kwargs.update(**CONFIG_DICT_STDALEXNET[config])
        super().__init__(**kwargs)


    @property
    def convs(self):
        """
        List of convolution layers

        """
        raise NotImplementedError


    def _structure_modifications(self, kernel_size=None, n_frozen_layers=0):

        # Replace the first convolution layer by an object inheriting from
        # 'base_modules.Conv2d'
        conv0 = self.features[0]
        kwargs = {}
        for argname in ARGS_CONV0:
            kwargs[argname] = getattr(conv0, argname)

        if kernel_size is not None:
            kwargs['kernel_size'] = kernel_size

        self.features[0] = base_modules.Conv2d(**kwargs)

        # Freeze layers, if required
        if n_frozen_layers > 0:
            convs = [self.features[i] for i in self.convs]
            cnn_toolbox.freeze_layers(convs, frozen_modules=n_frozen_layers)


class AlexNet(_BaseStdAlexNetMixin, models.AlexNet):
    """
    Standard AlexNet.

    Parameters
    ----------
    config (str, default='Std')
        Can be one of:
        - 'Std' (default): standard AlexNet;
        - 'Ks20': the kernel size is set to (20, 20);
        - 'Frozen': the first convolution layer is frozen during training.

    """
    @property
    def convs(self):
        return [0, 3, 6, 8, 10]


class ZhangAlexNet(_BaseStdAlexNetMixin, _BaseZhangAlexNet):
    """
    Antialiased AlexNet.

    Parameters
    ----------
    config (str, default='Std')
        Can be one of:
        - 'Std' (default): standard AlexNet;
        - 'Ks20': the kernel size is set to (20, 20);
        - 'Frozen': the first convolution layer is frozen during training.

    """
    @property
    def convs(self):
        return [0, 5, 9, 11, 13]


#===============================================================================
# Base class for all WPT AlexNets
#===============================================================================

def _get_maxpool(
        n_channels=64, has_blurpool=False, blurfilt_size=None, **kwargs
):
    # ReLU is placed before the max pooling layer.
    if not has_blurpool:
        out = [
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1)
        ]
    else:
        blurpool = model_toolbox.get_blurpool_constructor(
            n_channels, kwargs, blurfilt_size=blurfilt_size
        )
        out = [
            nn.ReLU(inplace=True),
            blurpool(n_channels, **kwargs),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1),
            blurpool(n_channels, **kwargs)
        ]

    return nn.Sequential(*out)


def _get_modulus(
        n_channels=64, stride=2, has_blurpool=False, blurfilt_size=None, **kwargs
):
    # Bias and ReLU are placed after the max pooling layer.
    out = [
        base_modules.Downsample2d(stride=stride),
        base_modules.Modulus(),
        base_modules.Bias(n_channels),
        nn.ReLU(inplace=True)
    ]
    if has_blurpool:
        blurpool = model_toolbox.get_blurpool_constructor(
            n_channels, kwargs, blurfilt_size=blurfilt_size
        )
        out.append(blurpool(n_channels, **kwargs))

    return nn.Sequential(*out)


def _structure_modifications0(
        net, antialiased=False, cgmod=False, blur_modulus=False,
        has_freelytrainedkernels=False, whichtypes=None,
        out_channels_freeconv=None, out_channels_maxpool=None,
        bias_wptblock=None, blurfilt_size=None, **kwargs
):
    module_list = list(net.features.children())

    # Create module replacing the first freely-trained convolution layer
    # Argument 'out_featmaps' is ignored if split_out_featmaps is provided
    if not antialiased:
        stride = 4 # Standard AlexNet
    else:
        stride = 2 # Antialiased AlexNet
    kwargs.update(in_channels=3, stride=stride, out_featmaps=64)
    if not has_freelytrainedkernels:
        if bias_wptblock is not None:
            kwargs.update(bias=bias_wptblock)
        wptlayer = building_blocks.WptBlock(**kwargs)
    else:
        kwargs.update(
            out_channels_freeconv=out_channels_freeconv,
            kernel_size_freeconv=11, padding=2,
            bias_wptblock=bias_wptblock
        )
        wptlayer = building_blocks.HybridConv2dWpt(**kwargs)

    module_list[0] = wptlayer

    # Create activation + pooling module
    kwargs_maxpool = {}
    if antialiased:
        kwargs_maxpool.update(
            has_blurpool=True, blurfilt_size=blurfilt_size
        )

    kwargs_modulus = {}
    if antialiased:
        if not blur_modulus: # Increase stride to match the final
                             # subsampling factor
            kwargs_modulus.update(stride=4)
        else:
            kwargs_modulus.update(
                has_blurpool=True, blurfilt_size=blurfilt_size
            )

    if cgmod:
        if out_channels_maxpool is None:
            # By default, the freely-trained channels are followed by max
            # pooling and the WPT channels are followed by modulus.
            out_channels_maxpool = out_channels_freeconv

        pooling_low = _get_maxpool(
            n_channels=out_channels_maxpool, **kwargs_maxpool
        )
        pooling_high = _get_modulus(
            n_channels=(64 - out_channels_maxpool), **kwargs_modulus
        )
        poolinglayer = building_blocks.Pooling(
            wptlayer.split_out_sections, whichtypes, pooling_low, pooling_high
        )

    else:
        poolinglayer = _get_maxpool(**kwargs_maxpool)

    # Replace actual activation + pooling by the custom module
    module_list.pop(1) # Remove ReLU
    if antialiased:
        module_list.pop(1) # Remove first BlurPool
    module_list.pop(1) # Remove MaxPool2d
    if antialiased:
        module_list.pop(1) # Remove second BlurPool

    module_list.insert(1, poolinglayer)

    net.features = nn.Sequential(*module_list)


class _BaseWptAlexNet(_BaseAlexNetMixin, models.AlexNet):
    def _structure_modifications(self, **kwargs):
        _structure_modifications0(self, **kwargs)


class _BaseWptZhangAlexNet(_BaseAlexNetMixin, _BaseZhangAlexNet):
    def _structure_modifications(self, **kwargs):
        _structure_modifications0(self, antialiased=True, **kwargs)


#===============================================================================
# WPT AlexNet based on the separable wavelet packet transform
#===============================================================================

class _WptAlexNetMixin:
    """
    This network is similar to a standard AlexNet, except that the first convolution
    layers are replaced by a wavelet packet transform (WPT) module with pseudo
    local cosine decomposition. See wcnn.cnn.building_blocks.WptBlock and
    wcnn.cnn.wpt.Wpt2d for more details.

    Parameters
    ----------
    wavelet (str, default=None)
        See dtcwt.coeffs.qshift for more details.

    backend (str, default='dtcwt')
        One of 'pywt' or 'dtcwt'. Name of the library from which to get the
        filters.

    padding_mode (str, default='zeros')
        Padding mode in the WPT layer. One of 'zeros' (default) or 'symmetric'.

    """
    def __init__(self, **kwargs):
        super().__init__(wpt_type='wpt2d', **kwargs)


class WptAlexNet(_WptAlexNetMixin, _BaseWptAlexNet):
    pass
class WptZhangAlexNet(_WptAlexNetMixin, _BaseWptZhangAlexNet):
    pass


#===============================================================================
# WPT AlexNet based on the dual-tree complex wavelet transform
#===============================================================================

# Dictionary of architectures for DtCwptAlexNet. Each key (of type str) denotes
# a specific configuration and the values are dictionaries of keyword arguments
# to get the desired configuration.
CONFIG_DICT_DTCWPTALEXNET = {}

# Keyword arguments which are common to all DT-CWPT-based models
_KWARGS0 = dict(wpt_type='dtcwpt2d', depth=3)

# Freely-trained channels
_OUT_CHANNELS_FREECONV = 32
_KWARGS_FREECONV = dict(
    has_freelytrainedkernels=True, out_channels_freeconv=_OUT_CHANNELS_FREECONV
)

# Initial color mix
_INIT_COLORMIX_WEIGHT = torch.tensor([[[[1.0036]], [[1.1630]], [[0.8334]]]])
_KWARGS_COLORMIX = dict(
    color_mixing_before_wpt=True, init_colormix_weight=_INIT_COLORMIX_WEIGHT
)

# Wavelet block
## Permute feature maps before filter selection, if required
_PERMUTE_FEATMAPS = [
      1,   2,  67,  65,  66,   3,   4,   5,  70,  71,   8,  10,  73,
     75,  76,  77,  78,  79,   6,   7,  68,  69,   9,  11,  72,  74,
     12,  13,  14,  15,  17,  18,  20,  21,  24,  26,  80,  83,  86,
     87,  89,  91,  92,  93,  94,  95,  33,  34,  36,  37,  40,  42,
     96,  99, 102, 103, 105, 107, 108, 109, 110, 111,  16,  19,  22,
     23,  25,  27,  28,  29,  30,  31,  81,  82,  84,  85,  88,  90,
     32,  35,  38,  39,  41,  43,  44,  45,  46,  47,  97,  98, 100,
    101, 104, 106
]

_SPLIT_IN_FEATMAPS = [6, 24, 64]
_SPLIT_OUT_FEATMAPS = [6, 18, 8]

## Number of groups for each scale (GCD between the number of input and outputs).
## Scale 3 has less groups since some filters are discarded.
_GROUPS = [6, 6, 4]

_KWARGS = dict(
    permute_featmaps=_PERMUTE_FEATMAPS,
    split_in_featmaps=_SPLIT_IN_FEATMAPS, split_out_featmaps=_SPLIT_OUT_FEATMAPS,
    groups=_GROUPS, **_KWARGS_FREECONV, **_KWARGS0
)

# WAlexNet (RMax)
# ==============================================================================
CONFIG_DICT_DTCWPTALEXNET.update({
    'Ft32': dict(
        real_part_only=True, **_KWARGS
    ),
    'Ft32Y': dict(
        real_part_only=True, **_KWARGS_COLORMIX, **_KWARGS
    ),
    'Ft32YBf2': dict(
        real_part_only=True, blurfilt_size=2, **_KWARGS_COLORMIX, **_KWARGS
    ),
    'Ft32YBf3': dict(
        real_part_only=True, blurfilt_size=3, **_KWARGS_COLORMIX, **_KWARGS
    ),
    'Ft32YBf4': dict(
        real_part_only=True, blurfilt_size=4, **_KWARGS_COLORMIX, **_KWARGS
    ),
    'Ft32YBf5': dict(
        real_part_only=True, blurfilt_size=5, **_KWARGS_COLORMIX, **_KWARGS
    ),
    'Ft32YBf6': dict(
        real_part_only=True, blurfilt_size=6, **_KWARGS_COLORMIX, **_KWARGS
    ),
    'Ft32YBf7': dict(
        real_part_only=True, blurfilt_size=7, **_KWARGS_COLORMIX, **_KWARGS
    )
})

# CWAlexNet (CMod)
# ==============================================================================
_WHICHTYPES = ['low'] + 3 * ['high']
_KWARGS_ACTIVATION = dict(
    cgmod=True, whichtypes=_WHICHTYPES
)
CONFIG_DICT_DTCWPTALEXNET.update({
    'Ft32_mod': dict(
        bias_wptblock=False, **_KWARGS_ACTIVATION, **_KWARGS
    ),
    'Ft32Y_mod': dict(
        bias_wptblock=False, **_KWARGS_COLORMIX,
        **_KWARGS_ACTIVATION, **_KWARGS
    ),
    'Ft32YBf2_mod': dict(
        bias_wptblock=False, blurfilt_size=2, **_KWARGS_COLORMIX,
        **_KWARGS_ACTIVATION, **_KWARGS
    ),
    'Ft32YBf3_mod': dict(
        bias_wptblock=False, blurfilt_size=3, **_KWARGS_COLORMIX,
        **_KWARGS_ACTIVATION, **_KWARGS
    ),
    'Ft32YBf4_mod': dict(
        bias_wptblock=False, blurfilt_size=4, **_KWARGS_COLORMIX,
        **_KWARGS_ACTIVATION, **_KWARGS
    ),
    'Ft32YBf5_mod': dict(
        bias_wptblock=False, blurfilt_size=5, **_KWARGS_COLORMIX,
        **_KWARGS_ACTIVATION, **_KWARGS
    ),
    'Ft32YBf6_mod': dict(
        bias_wptblock=False, blurfilt_size=6, **_KWARGS_COLORMIX,
        **_KWARGS_ACTIVATION, **_KWARGS
    ),
    'Ft32YBf7_mod': dict(
        bias_wptblock=False, blurfilt_size=7, **_KWARGS_COLORMIX,
        **_KWARGS_ACTIVATION, **_KWARGS
    )
})

class _DtCwptAlexNetMixin:
    """
    This network is similar to a standard AlexNet, except that some layers are
    replaced by others:
        - the first convolution layer is replaced by an module implementing the
          2D dual-tree complex wavelet packet transform. For more details, see
                > wcnn.cnn.building_blocks.HybridConv2dWpt;
                > wcnn.cnn.building_blocks.WptBlock;
                > wcnn.cnn.wpt.DtCwpt2d.
        - the following activation layers (ReLU and MaxPool2d) are replaced by
          a non-linear operator whose behavior depend on the chosen configuration.

    Parameters
    ----------
    config (str)
        String describing the network's configuration. Possible values are given
        by the keys of the global variable
                wcnn.cnn.models.alexnet.CONFIG_DICT_DTCWPTALEXNET.

    wavelet (str, default=None)
        See dtcwt.coeffs.qshift for more details.

    backend (str, default='dtcwt')
        One of 'pywt' or 'dtcwt'. Name of the library from which to get the
        filters.

    padding_mode (str, default='zeros')
        Padding mode in the WPT layer. One of 'zeros' (default) or 'symmetric'.

    """
    def __init__(self, config, **kwargs):

        kwargs.update(**CONFIG_DICT_DTCWPTALEXNET[config])
        super().__init__(**kwargs)


    def load_state_dict(
            self, state_dict, backward_compatibility=False, modulus=False, **kwargs
    ):
        # Patch following commits nb 1ef262b322fab5ee4e0a31a8d181aba4748c34d4
        # and 4bd8683ec343b89381a0bd4f5b9299a6687eb946
        if backward_compatibility:
            keyvals = []
            for k, val in state_dict.items():
                klist = k.split('.')

                if klist[0] == 'features':
                    layer = int(klist[1])
                    if layer == 0:
                        if klist[2] == '0':
                            klist[2] = 'freeconv'
                        elif klist[2] == '1':
                            klist[2] = 'wptblock'
                    elif not modulus:
                        if layer >= 3: # After ReLU and MaxPool
                            klist[1] = str(layer - 1)
                    else:
                        if layer == 2: # Bias
                            # Low-pass filters
                            if klist[-1] == 'bias':
                                val0 = val[:32]
                                keyvals.append(('features.0.freeconv.bias', val0))
                            # High-pass filters
                            val = val[32:]
                            klist = [klist[0], '1', 'high', '2', klist[-1]]
                        elif layer >= 4: # After pooling, bias and ReLU
                            klist[1] = str(layer - 2)

                k = '.'.join(klist)
                keyvals.append((k, val))
            state_dict = OrderedDict(keyvals)

        return super().load_state_dict(state_dict, **kwargs)



class DtCwptAlexNet(_DtCwptAlexNetMixin, _BaseWptAlexNet):
    pass
class DtCwptZhangAlexNet(_DtCwptAlexNetMixin, _BaseWptZhangAlexNet):
    pass
