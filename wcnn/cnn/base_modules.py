import torch
from torch import nn

from . import cnn_toolbox

class ConvKernelMixin:
    """
    Mixin class for modules performing a series of 2D convolutions. This is
    equivalent to propagate the input images through a single convolution layer.

    """
    @property
    def resulting_kernel(self):
        """
        Resulting convolution kernel.

        """
        with torch.no_grad():
            out, _ = self.get_resulting_kerstride()
        return out

    @property
    def resulting_stride(self):
        """
        Resulting stride.

        """
        try:
            out = self._stride
        except AttributeError:
            with torch.no_grad():
                _, out = self.get_resulting_kerstride()
        if not isinstance(out, tuple):
            if isinstance(out, int):
                out = (out, out)
            else:
                raise ValueError("Resulting stride must be an integer or a "
                                 "tuple.")
        return out


    def get_in_channels(self):
        """
        Returns the number of input channels

        """
        try:
            out = self._in_channels
        except AttributeError as exc:
            raise NotImplementedError from exc
        return out


    def get_out_channels(self):
        """
        Returns the number of output channels

        """
        try:
            out = self._out_channels
        except AttributeError as exc:
            raise NotImplementedError from exc
        return out


    def get_resulting_kerstride(self):
        """
        Returns the resulting convolution kernel and stride. These properties
        can be accessed through 'resulting_kernel' and 'stride' data descriptors.

        """
        raise NotImplementedError


class UnbiasedPointwiseLinearMixin(ConvKernelMixin):
    """
    Mixin class for unbiased, undecimated 1x1 convolutions.

    Paramters
    ---------
    in_channels (int)

    """
    def __init__(self, in_channels, *args, **kwargs):
        self._in_channels = in_channels
        self._stride = 1
        super().__init__(*args, **kwargs)

    def get_resulting_kerstride(self):
        ker = cnn_toolbox.get_equivalent_convkernel(
            self.forward, self._in_channels
        )
        stride = 1
        return ker, stride


class GroupRGBChannels(UnbiasedPointwiseLinearMixin, nn.Module):
    """
    Reorganize input channels with the following hierarchy:
        1/ wavelet packet filters;
        2/ RGB input channels;
        3/ real and imaginary parts (if applicable).


    Example. Let L denote the number of wavelet packet filters. Input
    channels are organized as follows. For any input tensor x of shape
    (n_imgs, in_channels, height, width), we have

        x[i] = [
            ------ R input channel ------
            x[i, 0], x[i, 1],       (1st wavelet packet filter)
            ...
            x[i, 2L-2], x[i, 2L-1], (Lth wavelet packet filter)

            ------ G input channel ------
            x[i, 2L], x[i, 2L+1],
            ...
            x[i, 4L-2], x[i, 4L-1],

            ------ B input channel ------
            x[i, 4L], x[i, 4L+1],
            ...
            x[i, 6L-2], x[i, 6L-1]
        ].

    After reorganization, the wavelet feature maps are arranged this way:

        y[i] = [
            ---- 1st wavelet packet filter ----
            y[i, 0], y[i, 1], (R channel)
            y[i, 2], y[i, 3], (G channel)
            y[i, 4], y[i, 5], (B channel)

            ...

            ---- Lth wavelet packet filter ----
            y[i, 6L-6], y[i, 6L-5],
            y[i, 6L-4], y[i, 6L-3],
            y[i, 6L-2], y[i, 6L-11]
        ]

    """
    def __init__(self, in_channels, n_colors, complex_featmaps=True):

        super().__init__(in_channels)
        self._n_colors = n_colors
        self._complex_featmaps = complex_featmaps


    def forward(self, x):

        n_imgs, in_channels, height, width = x.shape
        assert in_channels == self._in_channels

        if self._complex_featmaps:
            # shape = (n_imgs, n_colors, n_wavefilters, 2, height, width)
            x = x.view(n_imgs, self._n_colors, -1, 2, height, width)

            # shape = (n_imgs, n_wavefilters, n_colors, 2, height, width)
            x = x.permute(0, 2, 1, 3, 4, 5)

        else:
            # shape = (n_imgs, n_colors, n_wavefilters, height, width)
            x = x.view(n_imgs, self._n_colors, -1, height, width)

            # shape = (n_imgs, n_wavefilters, n_colors, height, width)
            x = x.permute(0, 2, 1, 3, 4)

        # shape = (n_imgs, in_channels, height, width)
        x = x.reshape(n_imgs, -1, height, width)

        return x


    def extra_repr(self):
        kwargs = dict(in_channels=self._in_channels, n_colors=self._n_colors)
        if self._complex_featmaps:
            kwargs.update(complex_featmaps=True)
        return cnn_toolbox.get_extra_repr(**kwargs)


class PermuteFeatmaps(UnbiasedPointwiseLinearMixin, nn.Module):

    def __init__(self, in_channels, in_featmaps, permutation_list):
        assert in_channels % in_featmaps == 0
        n_channels_per_featmaps = in_channels // in_featmaps
        super().__init__(in_channels)
        self._in_featmaps = in_featmaps
        self._permutation_list = permutation_list
        self._out_channels = n_channels_per_featmaps * len(permutation_list)

    def forward(self, x):
        n_imgs, in_channels, height, width = x.shape
        assert in_channels == self._in_channels
        x = x.view(n_imgs, self._in_featmaps, -1, height, width)
        x = x[:, self._permutation_list]
        x = x.reshape(n_imgs, -1, width, height)
        return x

    def extra_repr(self):
        kwargs = dict(
            in_channels=self._in_channels, out_channels=self._out_channels
        )
        return cnn_toolbox.get_extra_repr(**kwargs)


class DuplicateFeatmaps(UnbiasedPointwiseLinearMixin, nn.Module):

    def __init__(self, in_channels, in_featmaps, dupl_factor):
        assert in_channels % in_featmaps == 0
        super().__init__(in_channels)
        self._in_featmaps = in_featmaps
        self._dupl_factor = dupl_factor
        self._out_channels = dupl_factor * in_channels

    def forward(self, x):
        n_imgs, in_channels, height, width = x.shape
        assert in_channels == self._in_channels
        x = x.view(n_imgs, self._in_featmaps, -1, height, width)
        x = x.repeat(1, 1, self._dupl_factor, 1, 1)
        x = x.reshape(n_imgs, -1, height, width)
        return x

    def extra_repr(self):
        kwargs = dict(
            in_channels=self._in_channels, in_featmaps=self._in_featmaps,
            dupl_factor=self._dupl_factor
        )
        return cnn_toolbox.get_extra_repr(**kwargs)


class MixDropout(UnbiasedPointwiseLinearMixin, nn.Dropout3d):

    def __init__(self, in_channels, in_featmaps, **kwargs):
        super().__init__(in_channels, **kwargs)
        assert in_channels % in_featmaps == 0
        self._out_channels = in_channels
        self._in_featmaps = in_featmaps
        self._n_channels_per_featmap = in_channels // in_featmaps


    def forward(self, x):
        return _forward_3d(
            super().forward, x, self._in_channels, self._in_featmaps
        )

    def get_resulting_kerstride(self):

        in_channels = self._in_channels
        self._in_channels = self._in_featmaps
        ker0, stride = super().get_resulting_kerstride()
        self._in_channels = in_channels
        ker = torch.zeros(in_channels, in_channels, 1, 1)
        for k in range(self._n_channels_per_featmap):
            ker[
                k::self._n_channels_per_featmap,
                k::self._n_channels_per_featmap
            ] = ker0

        return ker, stride


    def extra_repr(self):
        out = super().extra_repr()
        out = ', '.join([out, "in_channels={}, in_featmaps={}".format(
            self._in_channels, self._in_featmaps
        )])
        return out


class MixModule(ConvKernelMixin, nn.ModuleList):
    """
    Module performing grouped 1x1 convolutions, with groups of varying size.
    This module is generally placed after a WPT decomposition.
    GIVE EXAMPLE.

    Parameters
    ----------
    in_channels (int)
    split_wavefilters (list of int)
    split_out_featmaps (list of int, default=None)
    out_featmaps (int, default=None)
    groups (int or list of int, default=None)
        For each bunch of channels, number of blocked connections from input
        channels to output channels. See torch.nn.Conv2d for more details.

    """
    def __init__(
            self, in_channels, split_wavefilters, split_out_featmaps=None,
            out_featmaps=None, groups=None, **kwargs
    ):
        super().__init__()

        n_splits = len(split_wavefilters)
        n_wavefilters = sum(split_wavefilters)

        # Number of input channels per feature map. For RGB images and complex-
        # valued feature maps, it is equal to 6.
        assert in_channels % n_wavefilters == 0
        in_channels_per_wavefilter = in_channels // n_wavefilters

        # Get input split sections for torch.split
        self._split_in_sections = [
            in_channels_per_wavefilter * n_wvfltrs \
                    for n_wvfltrs in split_wavefilters
        ]

        # Other private attributes
        self._in_channels = in_channels
        self._stride = 1

        # Check output split sections
        if split_out_featmaps is not None:
            assert len(split_out_featmaps) == n_splits
            self._out_featmaps = sum(split_out_featmaps)
        elif out_featmaps is not None: # Split output feature maps evenly
            assert out_featmaps % n_splits == 0
            self._out_featmaps = out_featmaps
            n_out_featmaps_per_group = out_featmaps // n_splits
            split_out_featmaps = n_splits * [n_out_featmaps_per_group]
        else:
            raise ValueError("Either 'split_out_featmaps' or 'out_featmaps' "
                             "must be specified.")

        self._split_out_sections = [
            in_channels_per_wavefilter * n_ftmps \
                    for n_ftmps in split_out_featmaps
        ]

        # Check number of groups, for each section
        if isinstance(groups, list):
            assert len(groups) == n_splits
        else:
            groups = n_splits * [groups]

        # Get list of convolution layers
        for in_chnls, n_wvfltrs, out_ftmps, grps in zip(
                self._split_in_sections, split_wavefilters, split_out_featmaps,
                groups
        ):
            layers = []

            if grps is not None:
                assert n_wvfltrs % grps == 0
                assert out_ftmps % grps == 0
                kwargs.update(groups=grps)

            layers.append(FeatmapMix(
                in_chnls, n_wvfltrs, out_ftmps, **kwargs
            ))

            if len(layers) > 1:
                self.append(Sequential(*layers))
            else:
                self.append(layers[0])

            # Remove 'groups' from kwargs (different for each bunch of channels)
            kwargs.pop('groups', None)


    @property
    def split_in_sections(self):
        return self._split_in_sections

    @property
    def split_out_sections(self):
        return self._split_out_sections

    @property
    def weights_l1reg(self):
        out = []
        for conv in self:
            if isinstance(conv, FeatmapMix):
                out.append(conv.weight)
            elif isinstance(conv, Sequential):
                out.append(conv[-1].weight)
            else:
                raise ValueError("Wrong type for feature map mixing layer.")
        return out


    def forward(self, x):

        # Forward-propagation of each sub-tensor
        inp = torch.split(x, self.split_in_sections, dim=1)
        out = []
        for conv, x0 in zip(self, inp):
            out.append(conv(x0))

        out = torch.cat(out, dim=1)

        return out


    def get_resulting_kerstride(self):

        # Get 1x1 convolution kernel (block diagonal matrix)
        kers = [conv.resulting_kernel[:, :, 0, 0] for conv in self]

        # Shape (out_channels, in_channels)
        ker = torch.block_diag(*kers)

        # Shape (out_channels, in_channels, 1, 1)
        ker = torch.unsqueeze(torch.unsqueeze(ker, -1), -1)
        stride = 1

        return ker, stride


    def get_out_channels(self):
        out = 0
        for conv in self:
            out += conv.get_out_channels()
        return out


class Sequential(ConvKernelMixin, nn.Sequential):
    """
    A standard sequential module equipped with ConvKernelMixin structure. Any
    submodule must inherit from ConvKernelMixin as well.

    """
    def get_in_channels(self):
        return self[0].get_in_channels()


    def get_out_channels(self):
        return self[-1].get_out_channels()


    def get_resulting_kerstride(self):

        list_of_kernels = []
        list_of_strides = []

        for layer in self:
            ker, stride = layer.get_resulting_kerstride()
            list_of_kernels.append(ker)
            list_of_strides.append(stride)

        out, resulting_stride = cnn_toolbox.convolve_kernels(
            self.get_in_channels(), list_of_kernels, list_of_strides
        )

        return out, resulting_stride


class Conv2d(ConvKernelMixin, nn.Conv2d):
    """
    A standard 2D convolution layer equipped with ConvKernelMixin structure.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set private attributes for methods 'get_in_channels' and
        # 'get_out_channels'
        self._in_channels = self.in_channels
        self._out_channels = self.out_channels

    @classmethod
    def import_from_parent(cls, inp):

        if not isinstance(inp, nn.Conv2d):
            raise TypeError
        args = [
            inp.in_channels, inp.out_channels, inp.kernel_size
        ]
        kwargs = dict(
            stride=inp.stride, padding=inp.padding, dilation=inp.dilation,
            groups=inp.groups, padding_mode=inp.padding_mode
        )
        if inp.bias is None:
            kwargs.update(bias=False)
        out = cls(*args, **kwargs)
        out.load_state_dict(inp.state_dict())

        return out


    def get_resulting_kerstride(self):

        ker0 = self.weight.detach()

        if self.groups > 1:
            ker = torch.zeros(
                self.out_channels, self.in_channels, *self.kernel_size,
                device=ker0.device
            )
            out_channels_per_group = self.out_channels // self.groups
            in_channels_per_group = self.in_channels // self.groups
            in_inf = 0
            in_sup = in_channels_per_group
            out_inf = 0
            out_sup = out_channels_per_group
            for _ in range(self.groups):
                ker[out_inf:out_sup, in_inf:in_sup] = ker0[out_inf:out_sup]
                in_inf = in_sup
                in_sup += in_channels_per_group
                out_inf = out_sup
                out_sup += out_channels_per_group
        else:
            ker = ker0

        return ker, self.stride


    def get_in_channels(self):
        return self.in_channels


class ComplexOutputConv2d(Conv2d):
    """
    Convolution layer with complex-valued kernels and outputs. The inputs and
    bias however remain real-valued. The real and imaginary parts are stored in
    separate channels.

    """
    def __init__(self, in_channels, out_featmaps, kernel_size, bias=True,
                 **kwargs):

        out_channels = 2 * out_featmaps
        super().__init__(
            in_channels, out_channels, kernel_size, bias=False, **kwargs
        )
        kwargs.pop('groups', None)
        if bias:
            self.biasmodule = Bias(out_featmaps, **kwargs)


    def forward(self, x):
        out = super().forward(x)
        if hasattr(self, 'biasmodule'):
            out[:, 0::2] = self.biasmodule(out[:, 0::2])
            out[:, 1::2] = self.biasmodule(out[:, 1::2])
        return out


class ComplexConv2d(Conv2d):
    """
    Convolution layer with complex-valued inputs and outputs. The real and
    imaginary parts are stored in separate channels.

    """
    def __init__(self, in_featmaps, out_featmaps, *args, real_part_only=False,
                 **kwargs):

        super().__init__(in_featmaps, out_featmaps, *args, **kwargs)

        # CAUTION: self._in_channels and self._out_channels differ from
        # self.in_channels and self.out_channels, respectively. The private
        # attributes provide the actual number of input and output channels
        # whereas the public attributes indicate the number of channels of the
        # embedded convolution layer, which correspond to the number of complex-
        # or real-valued feature maps.
        self._in_channels = 2 * in_featmaps
        if not real_part_only:
            self._out_channels = 2 * out_featmaps
        else:
            self._out_channels = out_featmaps

        self._real_part_only = real_part_only


    def forward(self, x):

        # Forward-propagation on even channels (real part of feature maps)
        out = super().forward(x[:, ::2])

        # If required, also perform forward-propagation on odd channels and
        # concatenate the two tensors with alternating channels.
        if not self._real_part_only:

            n_imgs = out.shape[0]
            height, width = out.shape[-2:]

            out_even = out
            out_odd = super().forward(x[:, 1::2])

            out = torch.zeros(
                n_imgs, self._out_channels, height, width, device=x.device
            )
            out[:, ::2] = out_even
            out[:, 1::2] = out_odd

        return out


    def get_resulting_kerstride(self):

        ker = torch.zeros(
            self._out_channels, self._in_channels, *self.kernel_size
        )
        ker0, stride = super().get_resulting_kerstride()
        if not self._real_part_only:
            # The embedded convolution kernel is used twice
            ker[0::2, 0::2] = ker0
            ker[1::2, 1::2] = ker0
        else:
            # Odd input channels are discarded
            ker[:, ::2] = ker0

        return ker, stride


    def get_in_channels(self):
        return 2 * super().get_in_channels()


    def extra_repr(self):

        out = super().extra_repr()
        if self._real_part_only:
            out = ', '.join([out, 'real_part_only=True'])

        return out


class FeatmapMix(UnbiasedPointwiseLinearMixin, nn.Conv3d):

    def __init__(self, in_channels, in_featmaps, out_featmaps, **kwargs):
        super().__init__(
            in_channels, in_featmaps, out_featmaps, kernel_size=1, stride=1,
            **kwargs
        )
        assert in_channels % in_featmaps == 0
        self._n_channels_per_featmap = in_channels // in_featmaps
        self._in_featmaps = in_featmaps
        self._out_featmaps = out_featmaps

    def forward(self, x):
        return _forward_3d(
            super().forward, x, self._in_channels, self._in_featmaps
        )

    def get_resulting_kerstride(self):
        # Remove bias to compute the resulting kernel
        bias = self.bias
        self.bias = None
        ker, stride = super().get_resulting_kerstride()
        self.bias = bias
        return ker, stride

    def get_out_channels(self):
        return self._n_channels_per_featmap * self._out_featmaps


class Downsample2d(nn.Module):

    def __init__(self, stride=2):
        if isinstance(stride, int):
            stride = (stride, stride)
        self._stride = stride
        super().__init__()

    def forward(self, x):
        s1, s2 = self._stride
        return x[:, :, ::s1, ::s2]

    def extra_repr(self):
        return cnn_toolbox.get_extra_repr(stride=self._stride)


class Modulus(nn.Module):

    def forward(self, x):
        """
        Parameters
        ----------
        x (torch.tensor)
            Tensor of shape (n_imgs, in_channels, width, height), where
            in_channels is expected to be even.

        """
        return cnn_toolbox.get_modulus(x)


class RealPartOnly(UnbiasedPointwiseLinearMixin, nn.Module):

    def __init__(self, in_channels, split_in_sections=None, real_part_only=True):

        super().__init__(in_channels)

        if split_in_sections is None:
            split_in_sections = [in_channels]
        n_sections = len(split_in_sections)
        if isinstance(real_part_only, list):
            assert len(real_part_only) == n_sections
        else:
            real_part_only = n_sections * [real_part_only]
        split_in_sections, real_part_only = cnn_toolbox.merge_sections(
            split_in_sections, real_part_only
        )
        self._split_in_sections = split_in_sections
        self._real_part_only = real_part_only
        self._split_out_sections = []
        for sis, rpo in zip(split_in_sections, real_part_only):
            if rpo:
                assert sis % 2 == 0
                self._split_out_sections.append(sis // 2)
            else:
                self._split_out_sections.append(sis)
        self._out_channels = sum(self._split_out_sections)

    @property
    def split_out_sections(self):
        return self._split_out_sections


    def forward(self, x):

        inp = torch.split(x, self._split_in_sections, dim=1)
        out = []
        for x0, rpo in zip(inp, self._real_part_only):
            if rpo:
                out.append(x0[:, ::2])
            else:
                out.append(x0)

        return torch.cat(out, dim=1)


    def extra_repr(self):

        if len(self._split_in_sections) > 1:
            splitsecs = []
            for n_channels, rpo in zip(
                    self._split_in_sections, self._real_part_only
            ):
                splitsecs.append('{} ({})'.format(n_channels, rpo))
            splitsecs = ', '.join(splitsecs)
            out = 'in_channels: {}'.format(splitsecs)
        else:
            out = 'in_channels={}'.format(self._in_channels)

        return out


class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size=3, stride=2):
        padding = cnn_toolbox.get_padding(stride, kernel_size)
        super().__init__(kernel_size, stride=stride, padding=padding)


class AvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size=3, stride=2):
        padding = cnn_toolbox.get_padding(stride, kernel_size)
        super().__init__(kernel_size, stride=stride, padding=padding)


class Bias(Conv2d):

    def __init__(self, in_channels, **kwargs):

        super().__init__(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=1, groups=in_channels, **kwargs
        )

        self._kwargs_extra_repr = kwargs

        # Set convolution kernel to identity and freeze it
        state_dict = self.state_dict()
        state_dict['weight'] = torch.ones(in_channels).view(in_channels, 1, 1, 1)
        self.load_state_dict(state_dict)
        self.weight.requires_grad = False


    def extra_repr(self):
        return cnn_toolbox.get_extra_repr(
            self.in_channels, **self._kwargs_extra_repr
        )


class BatchNormOnGaborModulus2d(nn.BatchNorm2d):
    r"""
    Applies Batch Normalization on the modulus of CGMod outputs, in the following
    fashion:

    .. math::

        y = \frac{x}{\sqrt{\frac12 \|x\|_2^2 + \epsilon}} \times |\gamma| + \beta,

    where :math:`x` denotes the output of a CGMod operator (modulus of zero-mean
    complex Gabor-like coefficients).

    Customized from `Piotr Bialecki's code <https://github.com/ptrblck/
    pytorch_misc/blob/master/batch_norm_manual.py>`__, commit nb
    31ac50c415f16cf7fec277dbdba72b9fb4d732d3.

    See also `this discussion thread <https://discuss.pytorch.org/t/
    how-does-batchnorm-keeps-track-of-running-mean/40084/3>`__.

    """
    @property
    def running_energy(self):
        return self.running_var

    @running_energy.setter
    def running_energy(self, val):
        self.running_var = val

    def forward(self, inp):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # Use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # Use exponential moving average
                    exponential_average_factor = self.momentum

        # Calculate running estimates
        if self.training:
            mean_energy = torch.sum(torch.linalg.norm(inp, dim=(2, 3))**2, dim=0)
            n = inp.numel() / inp.size(1)
            mean_energy /= n
            with torch.no_grad():
                # Update running_energy (no bias correction, unlike ptrblck)
                self.running_energy = exponential_average_factor * mean_energy \
                    + (1 - exponential_average_factor) * self.running_energy
        else:
            mean_energy = self.running_energy

        out = inp / (torch.sqrt(mean_energy[None, :, None, None] / 2 + self.eps))
        if self.affine:
            out = out * torch.abs(self.weight[None, :, None, None]) \
                + self.bias[None, :, None, None]

        return out


class ParallelConv2d(ConvKernelMixin, nn.ModuleDict):

    def __init__(self, in_channels, stride, convdict):

        if not isinstance(stride, tuple):
            stride = (stride, stride)
        out_channels = 0
        for _, conv in convdict.items():
            assert conv.get_in_channels() == in_channels
            assert conv.resulting_stride == stride

        super().__init__(convdict)

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride


    def forward(self, x):

        out = []
        for _, conv in self.items():
            out.append(conv(x))
        # Crop feature maps before concatenation
        cnn_toolbox.crop_featmaps_before_concat(out)

        return torch.cat(out, dim=1)


    def get_resulting_kerstride(self):

        out = []
        for _, conv in self.items():
            out.append(conv.resulting_kernel)
        # Crop kernels before concatenation
        cnn_toolbox.pad_featmaps_before_concat(out)

        return torch.cat(out, dim=0), self._stride


class DistributedDataParallel(nn.parallel.DistributedDataParallel):
    @property
    def weights_l1reg(self):
        return self.module.weights_l1reg


def _forward_3d(func, x, in_channels, in_featmaps):

    n_imgs, n_channels, width, height = x.shape
    assert n_channels == in_channels

    x = x.view(n_imgs, in_featmaps, -1, width, height)
    x = func(x)
    x = x.view(n_imgs, -1, width, height)

    return x
