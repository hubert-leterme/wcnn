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

def _get_activation_pooling(
        which, n_channels=64, blurfilt_size=None, use_adaptive_blurpool=False, **kwargs
):
    if which =="max":
        out = [
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
    elif which == "maxblur":
        blurpool = model_toolbox.get_blurpool_constructor(
            n_channels, kwargs, blurfilt_size=blurfilt_size,
            use_adaptive_blurpool=use_adaptive_blurpool
        )
        out = [
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            blurpool(n_channels, **kwargs)
        ]
    elif which == "mod":
        out = [
            base_modules.Downsample2d(),
            base_modules.Modulus(),
            base_modules.BatchNormOnGaborModulus2d(n_channels),
            nn.ReLU(inplace=True)
        ]
    else:
        raise ValueError("Positional argument must be one of 'max', 'maxblur' or 'mod'.")

    return nn.Sequential(*out)


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


    def _structure_modifications(
            self, use_wpt_block=False,
            which_activation_freechannels=None,
            which_activation_gaborchannels=None,
            use_adaptive_blurpool=False,
            has_freelytrainedkernels=False, whichtypes=None,
            freeze_first_layer=False, out_channels_freeconv=None,
            blurfilt_size=None, **kwargs
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
                    kernel_size_freeconv=7, padding=3
                )
                wptlayer = building_blocks.HybridConv2dWpt(**kwargs)

            self.conv1 = wptlayer

            # Create activation + pooling module
            kwargs_activation_pooling = dict(
                blurfilt_size=blurfilt_size,
                use_adaptive_blurpool=use_adaptive_blurpool
            )
            if which_activation_freechannels == which_activation_gaborchannels:
                poolinglayer = _get_activation_pooling(
                    which_activation_freechannels, **kwargs_activation_pooling
                )
            else:
                # Freely-trained channels
                pooling_low = _get_activation_pooling(
                    which_activation_freechannels, n_channels=out_channels_freeconv,
                    **kwargs_activation_pooling
                )
                # WPT channels
                pooling_high = _get_activation_pooling(
                    which_activation_gaborchannels, n_channels=(64 - out_channels_freeconv),
                    **kwargs_activation_pooling
                )
                poolinglayer = building_blocks.Pooling(
                    wptlayer.split_out_sections, whichtypes, pooling_low, pooling_high
                )

            # Batch norm and ReLU are in the pooling layer
            self.bn1 = nn.Identity()
            self.relu = nn.Identity()
            self.maxpool = poolinglayer


        if freeze_first_layer:
            cnn_toolbox.freeze_layers([self.conv1], frozen_modules=1)


    def _get_list_of_kwargs_net(self):
        return ['norm_layer']


    def _keep_kwargs_for_structure_modifications(self):
        return ['blurfilt_size']


    def _replace_last_layer(self, num_classes):
        old_fc = self.fc
        self.fc = self._new_last_layer(old_fc, num_classes)


class ResNet(_BaseResNetMixin, models.ResNet):
    pass


class _BaseZhangZouResNetMixin:

    def __init__(
            self, *args, blurfilt_size=None, **kwargs
    ):
        if blurfilt_size is not None:
            kwargs.update(filter_size=blurfilt_size)
        super().__init__(*args, **kwargs)


class _BaseZhangResNet(_BaseZhangZouResNetMixin, zhangmodels.ResNet):
    pass

class _BaseZouResNet(_BaseZhangZouResNetMixin, zoumodels.ResNet):
    pass


class _BaseAntialiasedResNetMixin(_BaseResNetMixin):

    def __init__(self, *args, blurfilt_size=3, **kwargs):
        super().__init__(*args, blurfilt_size=blurfilt_size, **kwargs)

    def _structure_modifications(
            self, use_wpt_block=False, correct_maxpool_kersize=True, **kwargs
    ):
        # Correct bug in Zhang's implementation (was it done on purpose?),
        # if requested. Change kernel_size=2 to kernel_size=3.
        if not use_wpt_block and correct_maxpool_kersize:
            self.maxpool[0] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        return super()._structure_modifications(use_wpt_block=use_wpt_block, **kwargs)

    def _get_list_of_kwargs_net(self):
        out = super()._get_list_of_kwargs_net() + ['blurfilt_size']
        return out


class ZhangResNet(_BaseAntialiasedResNetMixin, _BaseZhangResNet):
    pass


class ZouResNet(_BaseAntialiasedResNetMixin, _BaseZouResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, pasa_group=8, **kwargs)


    def _get_list_of_kwargs_net(self):
        out = super()._get_list_of_kwargs_net() + ['pasa_group']
        return out


def _base_resnet(
        constr, nlayers, **kwargs
):
    if constr == ResNet:
        basicblock = models.resnet.BasicBlock
        bottleneck = models.resnet.Bottleneck
    elif constr == ZhangResNet:
        basicblock = zhangmodels.resnet.BasicBlock
        bottleneck = zhangmodels.resnet.Bottleneck
    elif constr == ZouResNet:
        basicblock = zoumodels.BasicBlock
        bottleneck = zoumodels.Bottleneck
    else:
        raise TypeError(constr.__name__)

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
    'Bf2': dict(blurfilt_size=2),
    'Bf3': dict(blurfilt_size=3),
    'Bf4': dict(blurfilt_size=4),
    'Bf5': dict(blurfilt_size=5),
    'Bf6': dict(blurfilt_size=6),
    'Bf7': dict(blurfilt_size=7),
    'Frozen': dict(freeze_first_layer=True),

    # Without bug correction (kernel size in maxpool)
    'StdK2': dict(correct_maxpool_kersize=False),
    'Bf3K2': dict(correct_maxpool_kersize=False, blurfilt_size=3),
    'Bf5K2': dict(correct_maxpool_kersize=False, blurfilt_size=5)
}

def _base_std_resnet(constr, nlayers, config="Std", **kwargs):
    kwargs.update(**CONFIG_DICT_STDRESNET[config])
    return _base_resnet(constr, nlayers, **kwargs)


def _resnet(nlayers, **kwargs):
    return _base_std_resnet(ResNet, nlayers, **kwargs)

def _zhang_resnet(nlayers, **kwargs):
    return _base_std_resnet(ZhangResNet, nlayers, **kwargs)

def _zou_resnet(nlayers, **kwargs):
    return _base_std_resnet(ZouResNet, nlayers, **kwargs)

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
    'Ft40YBf2Ex': dict(
        blurfilt_size=2, **_KWARGS_COLORMIX, **_KWARGS_RGPOOL_EXCLUDE_EDGES
    ),
    'Ft40YBf3Ex': dict(
        blurfilt_size=3, **_KWARGS_COLORMIX, **_KWARGS_RGPOOL_EXCLUDE_EDGES
    ),
    'Ft40YBf4Ex': dict(
        blurfilt_size=4, **_KWARGS_COLORMIX, **_KWARGS_RGPOOL_EXCLUDE_EDGES
    ),
    'Ft40YBf5Ex': dict(
        blurfilt_size=5, **_KWARGS_COLORMIX, **_KWARGS_RGPOOL_EXCLUDE_EDGES
    ),
    'Ft40YBf6Ex': dict(
        blurfilt_size=6, **_KWARGS_COLORMIX, **_KWARGS_RGPOOL_EXCLUDE_EDGES
    ),
    'Ft40YBf7Ex': dict(
        blurfilt_size=7, **_KWARGS_COLORMIX, **_KWARGS_RGPOOL_EXCLUDE_EDGES
    ),
})

# CWResNet (CGMod)
# ==============================================================================
_WHICHTYPES = ['low', 'high', 'high'] # Scale 0, scale 1 and scale 2
_WHICHTYPES_EXCLUDE_EDGES = ['low'] + 5 * ['high']
_KWARGS_CGMOD = dict(
    which_activation_gaborchannels="mod", whichtypes=_WHICHTYPES, **_KWARGS
)
_KWARGS_CGMOD_EXCLUDE_EDGES = dict(
    which_activation_gaborchannels="mod", whichtypes=_WHICHTYPES_EXCLUDE_EDGES,
    **_KWARGS_EXCLUDE_EDGES
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
    'Ft40YBf2Ex_mod': dict(
        blurfilt_size=2, **_KWARGS_COLORMIX, **_KWARGS_CGMOD_EXCLUDE_EDGES
    ),
    'Ft40YBf3Ex_mod': dict(
        blurfilt_size=3, **_KWARGS_COLORMIX, **_KWARGS_CGMOD_EXCLUDE_EDGES
    ),
    'Ft40YBf4Ex_mod': dict(
        blurfilt_size=4, **_KWARGS_COLORMIX, **_KWARGS_CGMOD_EXCLUDE_EDGES
    ),
    'Ft40YBf5Ex_mod': dict(
        blurfilt_size=5, **_KWARGS_COLORMIX, **_KWARGS_CGMOD_EXCLUDE_EDGES
    ),
    'Ft40YBf6Ex_mod': dict(
        blurfilt_size=6, **_KWARGS_COLORMIX, **_KWARGS_CGMOD_EXCLUDE_EDGES
    ),
    'Ft40YBf7Ex_mod': dict(
        blurfilt_size=7, **_KWARGS_COLORMIX, **_KWARGS_CGMOD_EXCLUDE_EDGES
    ),
})

# Ablations
# ==============================================================================
_WHICHTYPES = ['low', 'high']
_KWARGS_ABL1_EXCLUDE_EDGES = dict(
    real_part_only=True, which_activation_gaborchannels="maxblur",
    whichtypes=_WHICHTYPES, **_KWARGS_EXCLUDE_EDGES
)
_KWARGS_ABL2_EXCLUDE_EDGES = dict(
    real_part_only=True, which_activation_gaborchannels="max",
    whichtypes=_WHICHTYPES, **_KWARGS_EXCLUDE_EDGES
)
CONFIG_DICT_DTCWPTRESNET.update({
    'Ft40YBf3Ex_abl1': dict(
        blurfilt_size=3, **_KWARGS_COLORMIX, **_KWARGS_ABL1_EXCLUDE_EDGES
    ), # "Addiditve" ablation; to be used with 'dtcwpt_resnet'
    'Ft40YBf3Ex_abl1ada': dict(
        blurfilt_size=3, use_adaptive_blurpool=True,
        **_KWARGS_COLORMIX, **_KWARGS_ABL1_EXCLUDE_EDGES
    ), # "Addiditve" ablation; to be used with 'dtcwpt_resnet'
    'Ft40YBf3Ex_abl2': dict(
        blurfilt_size=3, **_KWARGS_COLORMIX, **_KWARGS_ABL2_EXCLUDE_EDGES
    ) # "Soustractive" ablation; to be used with 'dtcwpt_zhang_resnet'
})


def _base_dtcwpt_resnet(constr, nlayers, config, **kwargs):
    kwargs.update(**CONFIG_DICT_DTCWPTRESNET[config])
    out = _base_resnet(
        constr, nlayers, use_wpt_block=True, wpt_type='dtcwpt2d',
        depth=2, stride=2, **kwargs
    )
    return out


def _dtcwpt_resnet(
        nlayers, config, which_activation_freechannels="max",
        which_activation_gaborchannels="max", **kwargs
):
    return _base_dtcwpt_resnet(
        ResNet, nlayers, config,
        which_activation_freechannels=which_activation_freechannels,
        which_activation_gaborchannels=which_activation_gaborchannels,
        **kwargs
    )

def _dtcwpt_zhang_resnet(
        nlayers, config, which_activation_freechannels="maxblur",
        which_activation_gaborchannels="maxblur", **kwargs
):
    return _base_dtcwpt_resnet(
        ZhangResNet, nlayers, config,
        which_activation_freechannels=which_activation_freechannels,
        which_activation_gaborchannels=which_activation_gaborchannels,
        **kwargs
    )

def _dtcwpt_zou_resnet(
        nlayers, config, which_activation_freechannels="maxblur",
        which_activation_gaborchannels="maxblur", **kwargs
):
    return _base_dtcwpt_resnet(
        ZouResNet, nlayers, config,
        which_activation_freechannels=which_activation_freechannels,
        which_activation_gaborchannels=which_activation_gaborchannels,
        use_adaptive_blurpool=True, **kwargs
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
