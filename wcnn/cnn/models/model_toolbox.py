from torch import nn
import antialiased_cnns as zhangnn
import models_lpf as zounn

from ... import toolbox, classifier
from .. import cnn_toolbox

class CustomModelMixin:
    """
    num_classes (int, default=1000)
    pretrained_model (str or torch.nn.Module, default=None)
    status_pretrained (str, default=None)
        Must be specified if 'pretrained_model' is a string.

    """
    def __init__(
            self, *args, num_classes=1000, pretrained_model=None,
            status_pretrained=None, **kwargs
    ):
        kwargs_net = toolbox.extract_dict(
            self._get_list_of_kwargs_net(), kwargs,
            keep=self._keep_kwargs_for_structure_modifications()
        )
        kwargs_loadpretrained = toolbox.extract_dict(
            classifier.LIST_OF_KWARGS_LOAD, kwargs
        )

        if pretrained_model is not None:
            self._use_pretrained_model = True
            super().__init__(*args, **kwargs_net)
        else:
            self._use_pretrained_model = False
            super().__init__(*args, num_classes=num_classes, **kwargs_net)

        try:
            self._structure_modifications(**kwargs)
        except TypeError:
            # Avoids "TypeError: _structure_modifications() got an unexpected
            # keyword argument ...". This occurs when
            # self._structure_modifications(**kwargs) does nothing (i.e., for
            # standard models), but I do not really understand why...
            remove_args = [
                "skip_pretrained_freeconv", "trainable_qmf", "blurfilt_size"
            ]
            for arg in remove_args:
                kwargs.pop(arg, None)
            self._structure_modifications(**kwargs)

        if pretrained_model is not None:
            if isinstance(pretrained_model, str):
                net = classifier.WaveCnnClassifier.load(
                    pretrained_model, status=status_pretrained, warn=False,
                    **kwargs_loadpretrained
                ).net_
            elif isinstance(pretrained_model, nn.Module):
                net = pretrained_model
            else:
                raise TypeError("Argument 'pretrained_model' must be of type "
                                "'str' or 'torch.nn.Module'.")

            self._partial_pretraining = False

            self.load_state_dict(net.state_dict())

            self._replace_last_layer(num_classes)


    @property
    def pretraining_status(self):

        if self._use_pretrained_model:
            if self._partial_pretraining:
                out = "partial initialization from a pretrained model"
            else:
                out = "full initialization from a pretrained model"
        else:
            out = "from scratch"

        return out


    @property
    def weights_l1reg(self):
        """
        Return a list of parameter weights to regularize during training.
        Typically, these are the trainable parameters associated to the feature
        map selection after DT-CWPT.

        """
        return self.first_conv_layer.weights_l1reg


    @property
    def first_conv_layer(self):
        """
        First convolution layer of the network (if applicable).

        """
        raise NotImplementedError


    def get_submodules(self, which):
        """
        Return a list of submodules.

        """
        list_of_kwargs = self._get_list_of_kwargs_submodules(which)
        out = []
        for kwargs in list_of_kwargs:
            out.append(cnn_toolbox.extract_subnet(self, **kwargs))
        return out


    def getlayer(self, name):
        """
        Access layer by its name (from method 'named_modules')

        """
        name = name.split('.')
        out = self
        for layer in name:
            out = getattr(out, layer)
        return out


    def setlayer(self, name, module):
        """
        Replace specific sublayer by another.

        """
        name = name.split('.')
        attrname = name[-1]
        name = '.'.join(name[:-1])

        setattr(self.getlayer(name), attrname, module)


    def _get_list_of_kwargs_net(self):
        """
        List of keyword arguments to be passed to the network's constructor.

        """
        return [] # By default, no extra keyword argument


    def _keep_kwargs_for_structure_modifications(self):
        """
        List of keyword arguments which must be used twice: once for the
        constructor of the parent class; and once for the wavelet block.

        """
        return [] # By default, discard all keyword arguments


    def _get_list_of_kwargs_submodules(self, which):
        """
        Get list of kwargs to be passed to 'cnn_toolbox.extract_subnet'.

        """
        raise NotImplementedError # To be overridden in children classes


    def _structure_modifications(self, **kwargs):
        """
        Modifications with respect to the standard model. Called during __init__.

        """
        # By default, do nothing. To be overridden in children classes.


    def _replace_last_layer(self, num_classes):
        raise NotImplementedError


    def _new_last_layer(self, old_last_layer, num_classes):
        return nn.Linear(
            in_features=old_last_layer.in_features, out_features=num_classes,
            bias=True
        )


def get_blurpool_constructor(
        n_channels, kwargs, blurfilt_size=None, use_adaptive_blurpool=False
):
    # 'kwargs' must be passed as a single argument, not a sequence
    # of keyword arguments (otherwise, not updated)
    kwargs.update(stride=2)
    if not use_adaptive_blurpool:
        blurpool = zhangnn.BlurPool
        if blurfilt_size is not None:
            kwargs.update(filt_size=blurfilt_size)
    else:
        blurpool = zounn.Downsample_PASA_group_softmax
        if blurfilt_size is not None:
            kwargs.update(kernel_size=blurfilt_size)
        if n_channels % 8 != 0:
            raise ValueError(
                "The number of output channels must be a multiple of 8."
            )
        group = n_channels // 8 # Group size = 8 channels
        kwargs.update(group=group)

    return blurpool
