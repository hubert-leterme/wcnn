from typing import OrderedDict
import torch
from torch import nn
from torchvision import models
import antialiased_cnns as zhangmodels
from models_lpf import resnet_pasa_group_softmax as zoumodels


from .. import cnn_toolbox, base_modules
from .. import building_blocks

from . import model_toolbox

#===============================================================================
# Base class
#===============================================================================

def _get_maxpool(
        n_channels=64, has_blurpool=False, blurfilt_size=None,
        use_adaptive_blurpool=False, **kwargs
):

    out = [
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True)
        ]

    if not has_blurpool:
        out += [
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
    else:
        blurpool = model_toolbox.get_blurpool_constructor(
            n_channels, kwargs, blurfilt_size=blurfilt_size,
            use_adaptive_blurpool=use_adaptive_blurpool
        )
        out += [
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            blurpool(n_channels, **kwargs)
        ]

    return nn.Sequential(*out)


def _get_modulus(n_channels=64):

    out = [
        base_modules.Downsample2d(),
        base_modules.Modulus(),
        base_modules.BatchNormOnGaborModulus2d(n_channels),
        nn.ReLU(inplace=True)
    ]

    return nn.Sequential(*out)


def _structure_modifications0(
        net, antialiased=False, use_adaptive_blurpool=False,
        cgmod=False, use_wpt_block=False,
        has_freelytrainedkernels=False, whichtypes=None,
        freeze_first_layer=False, out_channels_freeconv=None,
        blurfilt_size=None, correct_maxpool_kersize=True,
        **kwargs
):
    if use_wpt_block:
        kwargs.update(
            in_channels=3, stride=2, out_featmaps=64, bias=False
        )
        if not has_freelytrainedkernels:
            wptlayer = building_blocks.WptBlock(**kwargs)
        else:
            kwargs.update(
                out_channels_freeconv=out_channels_freeconv,
                kernel_size_freeconv=7,
                padding=3
            )
            wptlayer = building_blocks.HybridConv2dWpt(**kwargs)

        net.conv1 = wptlayer

        # Create activation + pooling module
        kwargs_maxpool = {}
        kwargs_modulus = {}
        if antialiased:
            kwargs_maxpool.update(
                has_blurpool=True, blurfilt_size=blurfilt_size,
                use_adaptive_blurpool=use_adaptive_blurpool
            )

        if cgmod:
            # Freely-trained channels
            pooling_low = _get_maxpool(
                n_channels=out_channels_freeconv, **kwargs_maxpool
            )
            # WPT channels
            pooling_high = _get_modulus(
                n_channels=(64 - out_channels_freeconv), **kwargs_modulus
            )
            poolinglayer = building_blocks.Pooling(
                wptlayer.split_out_sections, whichtypes, pooling_low, pooling_high
            )

        else:
            poolinglayer = _get_maxpool(**kwargs_maxpool)

        # Batch norm and ReLU are in the pooling layer
        net.bn1 = nn.Identity()
        net.relu = nn.Identity()
        net.maxpool = poolinglayer

    else:
        # Correct bug in Zhang's implementation (was it done on purpose?),
        # if requested. Change kernel_size=2 to kernel_size=3.
        if antialiased and correct_maxpool_kersize:
            net.maxpool[0] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


    if freeze_first_layer:
        cnn_toolbox.freeze_layers([net.conv1], frozen_modules=1)


class _BaseResNetMixin(model_toolbox.CustomModelMixin):

    @property
    def first_conv_layer(self):
        return self.conv1


    def load_state_dict(
            self, state_dict, backward_compatibility=False, modulus=False, **kwargs
    ):
        # Patch following commits nb ac75c73d77fc223ce2daecba285f5111bb41489f
        # and 4bd8683ec343b89381a0bd4f5b9299a6687eb946
        if backward_compatibility:
            keyvals = []
            for k, val in state_dict.items():
                klist = k.split('.')

                if klist[0] == 'conv1':
                    if klist[1] == '0':
                        klist[1] = 'freeconv'
                    elif klist[1] == '1':
                        klist[1] = 'wptblock'
                elif klist[0] == 'bn1':
                    if not modulus:
                        klist[0] = 'maxpool.0'
                    else:
                        klist[0] = 'maxpool'

                k = '.'.join(klist)
                keyvals.append((k, val))
            state_dict = OrderedDict(keyvals)

        return super().load_state_dict(state_dict, **kwargs)


    def _structure_modifications(self, **kwargs):
        _structure_modifications0(self, **kwargs)


    def _get_list_of_kwargs_net(self):
        return ['norm_layer']


    def _replace_last_layer(self, num_classes):
        old_fc = self.fc
        self.fc = self._new_last_layer(old_fc, num_classes)


class ResNet(_BaseResNetMixin, models.ResNet):
    pass


class _BaseAntialiasedResNetMixin(_BaseResNetMixin):

    def __init__(self, *args, filter_size=4, **kwargs):
        super().__init__(*args, filter_size=filter_size, **kwargs)

    def _structure_modifications(self, **kwargs):
        super()._structure_modifications(antialiased=True, **kwargs)

    def _get_list_of_kwargs_net(self):
        out = super()._get_list_of_kwargs_net() + ['filter_size']
        return out


class ZhangResNet(_BaseAntialiasedResNetMixin, zhangmodels.ResNet):
    pass


class ZouResNet(_BaseAntialiasedResNetMixin, zoumodels.ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, pasa_group=8, **kwargs)


    def _structure_modifications(self, **kwargs):
        super()._structure_modifications(use_adaptive_blurpool=True, **kwargs)


    def _get_list_of_kwargs_net(self):
        out = super()._get_list_of_kwargs_net() + ['pasa_group']
        return out


def _base_resnet(
        nlayers, antialiased=False, use_adaptive_blurpool=False, **kwargs
):
    if not antialiased:
        constr = ResNet
        basicblock = models.resnet.BasicBlock
        bottleneck = models.resnet.Bottleneck
    elif not use_adaptive_blurpool:
        constr = ZhangResNet
        basicblock = zhangmodels.resnet.BasicBlock
        bottleneck = zhangmodels.resnet.Bottleneck
    else:
        constr = ZouResNet
        basicblock = zoumodels.BasicBlock
        bottleneck = zoumodels.Bottleneck

    if nlayers == 18:
        block = basicblock
        layers = [2, 2, 2, 2]
    elif nlayers == 34:
        block = basicblock
        layers = [3, 4, 6, 3]
    elif nlayers == 50:
        block = bottleneck
        layers = [3, 4, 6, 3]
    elif nlayers == 101:
        block = bottleneck
        layers = [3, 4, 23, 3]
    else:
        raise NotImplementedError

    return constr(block, layers, **kwargs)


#===============================================================================
# Standard and antialiased ResNet-34
#===============================================================================

CONFIG_DICT_STDRESNET = {
    # Standard model
    'Std': {},
    'Bf3': dict(filter_size=3),
    'Bf5': dict(filter_size=5),
    'Frozen': dict(freeze_first_layer=True),

    # Without bug correction (kernel size in maxpool)
    'StdK2': dict(correct_maxpool_kersize=False),
    'Bf3K2': dict(correct_maxpool_kersize=False, filter_size=3),
    'Bf5K2': dict(correct_maxpool_kersize=False, filter_size=5)
}

def _base_std_resnet(nlayers, config="Std", **kwargs):
    kwargs.update(**CONFIG_DICT_STDRESNET[config])
    return _base_resnet(nlayers, **kwargs)


def _resnet(nlayers, **kwargs):
    return _base_std_resnet(nlayers, antialiased=False, **kwargs)

def _zhang_resnet(nlayers, **kwargs):
    return _base_std_resnet(nlayers, antialiased=True, **kwargs)

def _zou_resnet(nlayers, **kwargs):
    return _base_std_resnet(
        nlayers, antialiased=True, use_adaptive_blurpool=True, **kwargs
    )

def resnet18(**kwargs):
    return _resnet(18, **kwargs)

def resnet34(**kwargs):
    return _resnet(34, **kwargs)

def resnet50(**kwargs):
    return _resnet(50, **kwargs)

def resnet101(**kwargs):
    return _resnet(101, **kwargs)

def zhang_resnet18(**kwargs):
    return _zhang_resnet(18, **kwargs)

def zhang_resnet34(**kwargs):
    return _zhang_resnet(34, **kwargs)

def zhang_resnet50(**kwargs):
    return _zhang_resnet(50, **kwargs)

def zhang_resnet101(**kwargs):
    return _zhang_resnet(101, **kwargs)

def zou_resnet18(**kwargs):
    return _zou_resnet(18, **kwargs)

def zou_resnet34(**kwargs):
    return _zou_resnet(34, **kwargs)

def zou_resnet50(**kwargs):
    return _zou_resnet(50, **kwargs)

def zou_resnet101(**kwargs):
    return _zou_resnet(101, **kwargs)


#===============================================================================
#===============================================================================
# WaveResNet-34
#===============================================================================
#===============================================================================

# Initial color mix
_INIT_COLORMIX_WEIGHT = torch.tensor([[[[0.9802]], [[1.3060]], [[0.7138]]]])
_KWARGS_COLORMIX = dict(
    color_mixing_before_wpt=True, init_colormix_weight=_INIT_COLORMIX_WEIGHT
)
# Split input feature maps by scale (low-pass filters are discarded)
_SPLIT_IN_FEATMAPS = [6, 24]
_SPLIT_IN_FEATMAPS_EXCLUDE_EDGES = [6, 4, 1, 4, 1]
# Number of groups for each scale
_GROUPS = [6, 6]
_GROUPS_EXCLUDE_EDGES = [6, 4, 1, 4, 1] # Each filter is its own group (deterministic)

# Wavelet block
# Permute feature maps before filter selection, if required
_PERMUTE_FEATMAPS = [
     1,  2, 19, 17, 18,  3,  5,  6, 20, 23,  9, 10, 24, 27, 13, 14, 28,
    31,  4,  7, 21, 22,  8, 11, 25, 26, 12, 15, 29, 30
]
_PERMUTE_FEATMAPS_EXCLUDE_EDGES = [
    1,  2, 19, 17, 18,  3,  5, 23, 10, 27, 31,  7, 21, 11, 26, 15
]

_KWARGS0 = dict(
    permute_featmaps=_PERMUTE_FEATMAPS, split_in_featmaps=_SPLIT_IN_FEATMAPS,
    groups=_GROUPS
)
_KWARGS0_EXCLUDE_EDGES = dict(
    permute_featmaps=_PERMUTE_FEATMAPS_EXCLUDE_EDGES,
    split_in_featmaps=_SPLIT_IN_FEATMAPS_EXCLUDE_EDGES,
    groups=_GROUPS_EXCLUDE_EDGES
)

# Keyword arguments for the freely-trained channels
_OUT_CHANNELS_FREECONV = 40
_KWARGS_FREECONV = dict(
    has_freelytrainedkernels=True, out_channels_freeconv=_OUT_CHANNELS_FREECONV
)

# Split output feature maps by scale
_SPLIT_OUT_FEATMAPS = [12, 12]
_SPLIT_OUT_FEATMAPS_EXCLUDE_EDGES = [12, 4, 2, 4, 2]

_KWARGS = dict(
    split_out_featmaps=_SPLIT_OUT_FEATMAPS, **_KWARGS_FREECONV, **_KWARGS0
)
_KWARGS_EXCLUDE_EDGES = dict(
    split_out_featmaps=_SPLIT_OUT_FEATMAPS_EXCLUDE_EDGES, **_KWARGS_FREECONV,
    **_KWARGS0_EXCLUDE_EDGES
)

CONFIG_DICT_DTCWPTRESNET = {}

# WResNet (RMax)
# ==============================================================================
_KWARGS_RGPOOL = dict(real_part_only=True, **_KWARGS)
_KWARGS_RGPOOL_EXCLUDE_EDGES = dict(real_part_only=True, **_KWARGS_EXCLUDE_EDGES)

CONFIG_DICT_DTCWPTRESNET.update({
    'Ft40': dict(
        **_KWARGS_RGPOOL
    ),
    'Ft40Y': dict(
        **_KWARGS_COLORMIX, **_KWARGS_RGPOOL
    ),
    'Ft40YBf3': dict(
        blurfilt_size=3, **_KWARGS_COLORMIX, **_KWARGS_RGPOOL
    ),
    'Ft40Ex': dict(
        **_KWARGS_RGPOOL_EXCLUDE_EDGES
    ),
    'Ft40YEx': dict(
        **_KWARGS_COLORMIX, **_KWARGS_RGPOOL_EXCLUDE_EDGES
    ),
    'Ft40YBf3Ex': dict(
        blurfilt_size=3, **_KWARGS_COLORMIX, **_KWARGS_RGPOOL_EXCLUDE_EDGES
    )
})

# CWResNet (CGMod)
# ==============================================================================
_WHICHTYPES = ['low', 'high', 'high'] # Scale 0, scale 1 and scale 2
_WHICHTYPES_EXCLUDE_EDGES = ['low'] + 5 * ['high']
_KWARGS_CGMOD = dict(
    cgmod=True, whichtypes=_WHICHTYPES, **_KWARGS
)
_KWARGS_CGMOD_EXCLUDE_EDGES = dict(
    cgmod=True, whichtypes=_WHICHTYPES_EXCLUDE_EDGES, **_KWARGS_EXCLUDE_EDGES
)
CONFIG_DICT_DTCWPTRESNET.update({
    'Ft40_mod': dict(
        **_KWARGS_CGMOD
    ),
    'Ft40Y_mod': dict(
        **_KWARGS_COLORMIX, **_KWARGS_CGMOD
    ),
    'Ft40YBf3_mod': dict(
        blurfilt_size=3, **_KWARGS_COLORMIX, **_KWARGS_CGMOD
    ),
    'Ft40Ex_mod': dict(
        **_KWARGS_CGMOD_EXCLUDE_EDGES
    ),
    'Ft40YEx_mod': dict(
        **_KWARGS_COLORMIX, **_KWARGS_CGMOD_EXCLUDE_EDGES
    ),
    'Ft40YBf3Ex_mod': dict(
        blurfilt_size=3, **_KWARGS_COLORMIX, **_KWARGS_CGMOD_EXCLUDE_EDGES
    )
})


def _base_dtcwpt_resnet(nlayers, config, **kwargs):
    kwargs.update(**CONFIG_DICT_DTCWPTRESNET[config])
    model_toolbox.update_kwargs_modelconstructor(kwargs)
    out = _base_resnet(
        nlayers, use_wpt_block=True, wpt_type='dtcwpt2d',
        depth=2, stride=2, **kwargs
    )
    return out


def _dtcwpt_resnet(nlayers, config, **kwargs):
    return _base_dtcwpt_resnet(nlayers, config, **kwargs)

def _dtcwpt_zhang_resnet(nlayers, config, **kwargs):
    return _base_dtcwpt_resnet(nlayers, config, antialiased=True, **kwargs)

def _dtcwpt_zou_resnet(nlayers, config, **kwargs):
    return _base_dtcwpt_resnet(
        nlayers, config, antialiased=True, use_adaptive_blurpool=True, **kwargs
    )


def dtcwpt_resnet18(config, **kwargs):
    return _dtcwpt_resnet(18, config, **kwargs)

def dtcwpt_resnet34(config, **kwargs):
    return _dtcwpt_resnet(34, config, **kwargs)

def dtcwpt_resnet50(config, **kwargs):
    return _dtcwpt_resnet(50, config, **kwargs)

def dtcwpt_resnet101(config, **kwargs):
    return _dtcwpt_resnet(101, config, **kwargs)

def dtcwpt_zhang_resnet18(config, **kwargs):
    return _dtcwpt_zhang_resnet(18, config, **kwargs)

def dtcwpt_zhang_resnet34(config, **kwargs):
    return _dtcwpt_zhang_resnet(34, config, **kwargs)

def dtcwpt_zhang_resnet50(config, **kwargs):
    return _dtcwpt_zhang_resnet(50, config, **kwargs)

def dtcwpt_zhang_resnet101(config, **kwargs):
    return _dtcwpt_zhang_resnet(101, config, **kwargs)

def dtcwpt_zou_resnet18(config, **kwargs):
    return _dtcwpt_zou_resnet(18, config, **kwargs)

def dtcwpt_zou_resnet34(config, **kwargs):
    return _dtcwpt_zou_resnet(34, config, **kwargs)

def dtcwpt_zou_resnet50(config, **kwargs):
    return _dtcwpt_zou_resnet(50, config, **kwargs)

def dtcwpt_zou_resnet101(config, **kwargs):
    return _dtcwpt_zou_resnet(101, config, **kwargs)
