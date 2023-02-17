import warnings
import math
import torch
from torch import nn
from torch.nn import functional as F
import pywt, dtcwt

from .. import cnn_toolbox, base_modules


class BaseDwt2d(base_modules.ConvKernelMixin, nn.Module):
    """
    Base class for 2D discrete wavelet transforms (WPT or DT-CWPT).
    The module is parametrized by one or two 1D low-pass CMF(s).

    Parameters
    ----------
    in_channels (int)
        Number of input channels.
    depth (int)
        Number of decomposition stages.
    stride (int)
        Subsampling level of the full decomposition.
    cmf_map (dict)
        Dictionary mapping each type of CMFs to its successors. Ex.: h >> [h, g].
    init_hcmftypes (list)
        List of initial low-pass CMF types.
    init_gcmftypes (list, defaul=None)
        List of initial high-pass CMF types.
    padding_mode (str, default='zeros')
        One of 'zeros' or 'reflect' ('periodic' may be implemented as well).
    wavelet (str, default=None)
        See dtcwt.coeffs.qshift for more details.
    backend (str, default='dtcwt')
        One of 'pywt' or 'dtcwt'. Name of the library from which to get the
        filters.
    n_filters (int, default=1)
        Number of 1D filter banks (1 for WPT, 2 for dual-tree).
    discard_featmaps (dict, default=None)
        Dictionary which gives, for some decomposition stages (> 0), a list of
        feature map indices to discard. Example:
            {
                1: [3],          // Indices between 0 and 3
                2: [4, 6, 8, 9]  // Indices between 0 and 11
            }.

    """
    def __init__(self, in_channels, depth, stride, cmf_map, init_hcmftypes,
                 init_gcmftypes=None, padding_mode='zeros',
                 wavelet=None, backend='dtcwt', n_filters=1,
                 discard_featmaps=None, verbose=False):

        super().__init__()

        # Private attributes
        self._in_channels = in_channels
        self._depth = depth
        self._stride = stride
        self._cmf_map = cmf_map
        self._init_hcmftypes = init_hcmftypes
        self._init_gcmftypes = init_gcmftypes
        self._wavelet = wavelet
        self._n_filters = n_filters
        self._n_2d_filters = n_filters**2
        self._discard_featmaps = discard_featmaps

        # Public attributes
        self.padding_mode = padding_mode
        self.verbose = verbose

        # Initialize list of CMF types for horizontal and vertical filters
        # (different for each stage of decomposition)
        self._vert_cmftypes = None
        self._hori_cmftypes = None

        # Initialize current depth as well as number of feature maps per input
        # channel and filter bank
        self._current_depth = 0
        self._n_featmaps = 1

        # Get filters
        hfilters, gfilters = _get_initial_filters(
            filter_size=None, wavelet=wavelet, backend=backend,
            n_filters=n_filters
        )
        filter_size = _check_consistency_and_get_filtersize(
            hfilters, gfilters, n_filters
        )
        self._filter_size = filter_size

        # The global stride is decomposed into a sequence of strides equal to 2,
        # or 1 when the desired subsampling factor is reached.
        depth0 = min(depth, math.ceil(math.log(stride, 2)))
        if 2**depth0 != stride:
            raise ValueError("Invalid value of 'stride'.")
        self._strides = depth0 * [2] + (depth - depth0) * [1]

        # Get list of dilation factors. If stride = 2 (subsampled WPT), then
        # the dilation factor do not change. If stride = 1 (stationary WPT),
        # then the dilation factor is multiplied by 2.
        self._dilations = []
        dilation = 1
        for s in self._strides:
            self._dilations.append(dilation)
            if s == 1:
                dilation *= 2

        # Cast filters into the highest size
        for i, hfilt in enumerate(hfilters):
            hfilters[i] = _check_filter_size_and_extend_filter(
                hfilt, filter_size
            )
        if gfilters is not None:
            for i, gfilt in enumerate(gfilters):
                gfilters[i] = _check_filter_size_and_extend_filter(
                    gfilt, filter_size
                )

        # Create trainable parameter and disable gradient if needed
        self._hfilters = nn.Parameter(
            torch.stack(hfilters), requires_grad=False
        )
        # Buffer (attributes that are not parameters but should also be sent
        # to device)
        if gfilters is not None:
            self.register_buffer('_gfilters', torch.stack(gfilters))
        else:
            self._gfilters = None

        # Assert that all values in _cmf_map have the same number of elements.
        first_cmf = list(self._cmf_map.keys())[0]
        self._dupl = len(self._cmf_map[first_cmf])
        for cmf in self._cmf_map:
            assert len(self._cmf_map[cmf]) == self._dupl

        # Compute number of output channels and padding
        out_channels, padding = self._outchannels_padding()
        self._out_channels = out_channels
        self._padding = padding


    @property
    def depth(self):
        return self._depth

    @property
    def strides(self):
        return self._strides

    @property
    def dilations(self):
        return self._dilations

    @property
    def cmf_dict(self):
        """
        Dictionary mapping each filter type to the corresponding filter.

        """
        raise NotImplementedError # Must be overridden in children classes

    @property
    def cmf_map(self):
        """
        Dictionary indicating, for each type of filter, the pair of low- and
        high-pass filters to apply at the next stage of decomposition.

        """
        return self._cmf_map

    @property
    def init_hcmftypes(self):
        """
        List of low-pass CMF types used to initialize the wavelet decomposition.

        """
        return self._init_hcmftypes

    @property
    def init_gcmftypes(self):
        """
        List of high-pass CMF types used to initialize the wavelet decomposition.

        """
        return self._init_gcmftypes

    @property
    def padding(self):
        return self._padding

    @property
    def wavelet(self):
        return self._wavelet

    @property
    def n_filters(self):
        return self._n_filters

    @property
    def n_2d_filters(self):
        return self._n_2d_filters

    @property
    def n_units(self):
        """
        Number of orientation groups. Equals 1 for WPT, 2 for DT-(RC)WPT layers.

        """
        raise NotImplementedError

    @property
    def discard_featmaps(self):
        if self._discard_featmaps is not None:
            out = self._discard_featmaps
        else:
            out = dict()
        return out

    @property
    def filter_size(self):
        """
        Size of the filters.
        If the original filters have different sizes, they are all casted to
        the maximum size.

        """
        return self._filter_size

    @property
    def learn_wavefilters(self):
        raise NotImplementedError

    @property
    def hfilters(self):
        """List of low-pass filters"""
        out = torch.split(self._hfilters, 1) # Shape of subtensors = (1, Q)
        out = [hfilt.flatten() for hfilt in out]
        return out

    @property
    def gfilters(self):
        """List of high-pass filters"""
        if self._gfilters is not None:
            out = torch.split(self._gfilters, 1) # Shape of subtensors = (1, Q)
            out = [gfilt.flatten() for gfilt in out]
        else:
            for hfilt in self.hfilters:
                out.append(_get_gfilter(hfilt))
        return out

    @property
    def scale_bounds(self):
        """
        Lists of bounds grouping feature map indices by scale. Example for
        DT-CWPT with 2 levels of decomposition, without discarding any filter:
            [2, 8, 32]
        The 32 next filters are ignored, since they follow the same pattern.

        """
        out = [1] # Initialize scale bounds (level 0)
        for j in range(self.depth):
            discard = self.discard_featmaps.get(j+1)
            if discard is None:
                discard = []
            out = _get_next_scale_bounds(out, discard)
        out.reverse()
        out = [self.n_filters * bound for bound in out]

        return out


    def forward(self, x):
        """
        Performs Q**2 separable wavelet transforms, Q being the number of 1D
        low-pass filters.

        Parameter
        ---------
        x (torch.Tensor of shape (n_imgs, in_channels, N, M))
            Input batch of images.

        Returns
        -------
        z (torch.Tensor)
            Tensor of shape (n_imgs, out_channels, N//(2**d), M//(2**d), where
            d (depth) is the number of iterations.
            Example for Q = 2, in_channels = 3 and d = 1, where ha and hb (resp.
            ga and gb) denote the low-pass filters (resp. the corresponding high-
            pass filters). The output feature maps are organized as follows, for
            any p < n_imgs:

        z[p] = stack [
            ----- aa filter bank -----
            x[p, 0]*(ha.ha), x[p, 0]*(ha.ga), x[p, 0]*(ga.ha), x[p, 0]*(ga.ga),
            x[p, 1]*(ha.ha), x[p, 1]*(ha.ga), x[p, 1]*(ga.ha), x[p, 1]*(ga.ga),
            x[p, 2]*(ha.ha), x[p, 2]*(ha.ga), x[p, 2]*(ga.ha), x[p, 2]*(ga.ga),

            ----- ab filter bank -----
            x[p, 3]*(ha.hb), x[p, 3]*(ha.gb), x[p, 3]*(ga.hb), x[p, 3]*(ga.gb),
            x[p, 4]*(ha.hb), x[p, 4]*(ha.gb), x[p, 4]*(ga.hb), x[p, 4]*(ga.gb),
            x[p, 5]*(ha.hb), x[p, 5]*(ha.gb), x[p, 5]*(ga.hb), x[p, 5]*(ga.gb),

            ----- ba filter bank -----
            ...

            ----- bb filter bank -----
            ...]

        """
        assert x.shape[1] == self._in_channels

        # Pad x
        x = self._pad(x)

        # Duplicate x (concatenation of itself along the channelwise dimension)
        x = self._duplicate_input(x)

        for st, dil in zip(self.strides, self.dilations):

            vweight, hweight = self._next_convkernels()
            x = cnn_toolbox.separable_conv2d(
                x, vweight, hweight, st, dilation=dil
            )

            # Discard feature maps if required
            discard = self.discard_featmaps.get(self._current_depth)
            if discard is not None:
                x = _del_featmaps(x, discard, self._n_featmaps)         

        self._reset_cmftypes()

        return x


    def get_resulting_kerstride(self):

        list_of_kernels = []
        list_of_strides = []
        list_of_dilations = []

        # Initial duplication of the input images
        list_of_kernels.append(cnn_toolbox.get_equivalent_convkernel(
            self._duplicate_input, self._in_channels
        ))
        list_of_strides.append(1)
        list_of_dilations.append(1)

        # Succession of separable convolutions
        for st, dil in zip(self.strides, self.dilations):
            vweight, hweight = self._next_convkernels()
            list_of_kernels += [vweight, hweight]
            list_of_strides += [(st, 1), (1, st)]
            list_of_dilations += [(dil, 1), (1, dil)]

            # Discard feature maps if required
            discard = self.discard_featmaps.get(self._current_depth)
            if discard is not None:
                args = [discard, self._n_featmaps]
                list_of_kernels.append(cnn_toolbox.get_equivalent_convkernel(
                    _del_featmaps, hweight.shape[0], *args
                ))
                list_of_strides.append(1)
                list_of_dilations.append(1)

        self._reset_cmftypes()

        ker, st = cnn_toolbox.convolve_kernels(
            self._in_channels, list_of_kernels, list_of_strides,
            list_of_dilations=list_of_dilations
        )
        return ker, st


    def extra_repr(self):

        args = [self._in_channels, self._out_channels]
        kwargs = dict(
            depth=self.depth, stride=self.resulting_stride, padding=self.padding
        )
        if self.resulting_stride == 1:
            kwargs.pop('stride')
        if self.wavelet is not None:
            kwargs.update(wavelet=self.wavelet)
        else:
            kwargs.update(filter_size=self.filter_size)
        if self.padding_mode != 'zeros':
            kwargs.update(padding_mode=self.padding_mode)

        out = cnn_toolbox.get_extra_repr(*args, **kwargs)

        return out


    def _outchannels_padding(self):

        kersizes = []
        for _ in range(self.depth):
            vweight, hweight = self._next_convkernels()
            kersizes.append((vweight.shape[-2], hweight.shape[-1]))

        # Update nb of feature maps at the last stage
        self._update_n_featmaps()
        out_channels = self._n_featmaps * self._in_channels * self._n_2d_filters
        if self.padding_mode == 'zeros':
            padding = cnn_toolbox.get_padding(
                self.strides, kersizes, self.dilations
            )
        else:
            padding = (0, 0)

        self._reset_cmftypes()

        return out_channels, padding


    def _pad(self, x):

        if self.padding_mode == 'zeros':
            padding_mode = 'constant'
        else:
            padding_mode = self.padding_mode

        pad_vert, pad_hori = cnn_toolbox.get_tuple(self.padding)

        x = F.pad(
            x, pad=(pad_hori, pad_hori, pad_vert, pad_vert), mode=padding_mode
        )
        return x


    def _duplicate_input(self, x):
        return torch.cat(self.n_2d_filters * [x], dim=1)


    def _next_convkernels(self):

        vweight = self._next_vertical_convkernels()
        hweight = self._next_horizontal_convkernels()

        # Update nb of feature maps if some are discarded
        self._update_n_featmaps()

        # Update number of feature maps for the next stage
        self._current_depth += 1
        self._n_featmaps *= self._dupl_factor(self._current_depth)**2

        return vweight, hweight


    def _next_vertical_convkernels(self):
        """
        Returns a weight tensor for torch.nn.functional.conv2d. It performs the
        vertical part of the separable 2D convolutions. The shape is
        (2KQ**2, 1, n, 1), where K denotes the number of input channels per 2D
        filter bank, Q denotes the number of 1D filters and n denotes the filter
        size. Then the number of output channels is equal to 2KQ**2 whereas the
        convolution kernels have shape (n, 1).

        For instance, we consider a list of 2 low-pass filters ha and hb. Let
        ga and gb denote the corresponding high-pass filters. We thus have four
        2D filter banks. Then, taking as input the RGB images, the output has
        the following pattern:

        W = stack [
            ----- aa filter bank -----
            ha, ga, (1st input channel)
            ha, ga, (2nd input channel)
            ha, ga, (3rd input channel)

            ----- ab filter bank -----
            ha, ga, (4th input channel)
            ha, ga, (5th input channel)
            ha, ga, (6th input channel)

            ----- ba filter bank -----
            hb, gb, (7th input channel)
            hb, gb, (8th input channel)
            hb, gb, (9th input channel)

            ----- bb filter bank -----
            hb, gb, (10th input channel)
            hb, gb, (11th input channel)
            hb, gb, (12th input channel)
        ]

        """
        if self._vert_cmftypes is None: # initialization
            dupl = self._in_channels * self.n_filters
            out = []
            for i, hfilt in enumerate(self.init_hcmftypes):
                if self.init_gcmftypes is not None:
                    gfilt = self.init_gcmftypes[i]
                    out += dupl * [hfilt, gfilt]
                else:
                    out += dupl * [hfilt]
            self._vert_cmftypes = out

        else: # get CMF types for the next level of decomposition

            # Duplicate CMFs (e.g.: [h, g] --> [h, h, g, g])
            dupl_next_stage = self._dupl_factor(self._current_depth)
            self._vert_cmftypes = _duplicate_cmftypes(
                self._vert_cmftypes, dupl_next_stage
            )

            # Discard CMFs (e.g.: [h, h, g, g] --> [h, h, g])
            discard = self.discard_featmaps.get(self._current_depth)
            if discard is not None:
                self._vert_cmftypes = _del_elts(
                    self._vert_cmftypes, discard, self._n_featmaps
                )

            # Get next CMFs (e.g.: [h, h, g] --> [{h, g}, {h, g}, {h, g}])
            self._vert_cmftypes = _get_next_cmftypes(
                self._vert_cmftypes, self.cmf_map
            )

        if self.verbose:
            _print_cmftypes(self._current_depth, 'vertical', self._vert_cmftypes)

        out = _get_convkernels(self._vert_cmftypes, self.cmf_dict)
        out = cnn_toolbox.unsqueeze_vertical_kernels(out)

        return out


    def _next_horizontal_convkernels(self):
        """
        Returns a weight tensor for torch.nn.functional.conv2d. It performs the
        horizontal part of the separable 2D convolutions. The shape is
        (4KQ**2, 1, 1, n), with same notations as _get_vertical_convkernels. Then
        the number of output channels is equal to 4KQ**2 whereas the convolution
        kernels have shape (1, n).

        By taking the same example as vert_cmftypes, the output has the following
        pattern:

        W = stack [
            ----- aa filter bank -----
            ha, ga, ha, ga, (1st input channel)
            ha, ga, ha, ga, (2nd input channel)
            ha, ga, ha, ga, (3rd input channel)

            ----- ab filter bank -----
            hb, gb, hb, gb, (4th input channel)
            hb, gb, hb, gb, (5th input channel)
            hb, gb, hb, gb, (6th input channel)

            ----- ba filter bank -----
            ha, ga, ha, ga, (7th input channel)
            ha, ga, ha, ga, (8th input channel)
            ha, ga, ha, ga, (9th input channel)

            ----- bb filter bank -----
            hb, gb, hb, gb, (10th input channel)
            hb, gb, hb, gb, (11th input channel)
            hb, gb, hb, gb, (12th input channel)
        ]

        """
        if self._hori_cmftypes is None: # initialization
            out = []
            for i, hfilt in enumerate(self.init_hcmftypes):
                if self.init_gcmftypes is not None:
                    gfilt = self.init_gcmftypes[i]
                    out += self._in_channels * [hfilt, gfilt, hfilt, gfilt]
                else:
                    out += self._in_channels * [hfilt]
            out *= self.n_filters
            self._hori_cmftypes = out

        else: # get CMF types for the next level of decomposition

            # Discard CMFs (e.g.: [h0, h0, g0, g0] --> [h0, h0, g0])
            discard = self.discard_featmaps.get(self._current_depth)
            if discard is not None:
                self._hori_cmftypes = _del_elts(
                    self._hori_cmftypes, discard, self._n_featmaps
                )

            # Get next CMFs (e.g.: [h0, h0, g0] --> [{h0, g0}, {h0, g0}, {h, g}])
            self._hori_cmftypes = _get_next_cmftypes(
                self._hori_cmftypes, self.cmf_map
            )

            # Duplicate CMFs (e.g.: [{h0, g0}, {h0, g0}, {h, g}] -->
            # [{h0, g0}, {h0, g0}, {h0, g0}, {h0, g0}, {h, g}, {h, g}])
            dupl_next_stage = self._dupl_factor(self._current_depth)
            self._hori_cmftypes = _duplicate_cmftypes(
                self._hori_cmftypes, dupl_next_stage, group=2
            )

        if self.verbose:
            _print_cmftypes(self._current_depth, 'horizontal', self._hori_cmftypes)

        out = _get_convkernels(self._hori_cmftypes, self.cmf_dict)
        out = cnn_toolbox.unsqueeze_horizontal_kernels(out)

        return out


    def _update_n_featmaps(self):
        # Update nb of feature maps if some are discarded
        discard = self.discard_featmaps.get(self._current_depth)
        if discard is not None:
            self._n_featmaps -= len(discard)


    def _dupl_factor(self, depth):
        """
        Duplication factor for a given decomposition stage. Equals 2 if both low-
        and high-pass filters are used at the current stage, 1 otherwise.

        """
        assert depth > 0
        return self._dupl


    def _reset_cmftypes(self):
        self._vert_cmftypes = None
        self._hori_cmftypes = None
        self._current_depth = 0
        self._n_featmaps = 1


def _check_filter_size_and_extend_filter(filt, filter_size):

    filter_size_0 = len(filt)
    if filter_size is not None and filter_size_0 < filter_size:
        # Adjust with zeros
        filt_extended = filter_size * [0.]
        beg = (filter_size - filter_size_0) // 2
        filt_extended[beg:beg + filter_size_0] = filt
        filt = filt_extended

    return filt


def _get_initial_filters(filter_size, wavelet, backend, n_filters):
    """
    Get low- and high-pass filters from pywt or dtcwt libraries, or initialize
    randomly.

    filter_size (int)
        Must be even, or None.
    wavelet (str)
        See dtcwt.coeffs.qshift for more details. Can also be None.
    backend (str, default='dtcwt')
        One of 'pywt' or 'dtcwt'. Name of the library from which to get the
        filters.
    n_filters (int)
        Number of low-pass filters to keep.

    """
    assert filter_size is None or filter_size % 2 == 0

    # Get initial values of the low-pass filter(s) h...
    if wavelet is not None:
        if backend == 'pywt':
            wavelet = pywt.Wavelet(wavelet)
            hfilters = [wavelet.dec_lo] # These filters are flipped w.r.t.
            gfilters = [wavelet.dec_hi] # the original filter

        elif backend == 'dtcwt':
            # hb is before ha; there seems to be a confusion in the notations
            # within the dtcwt package.
            hb, ha, _, _, gb, ga, _, _ = dtcwt.coeffs.qshift(wavelet)
            hfilters = [ha, hb]
            gfilters = [ga, gb]

        else:
            raise ValueError("Unknown backend.")

        hfilters = hfilters[:n_filters]
        gfilters = gfilters[:n_filters]

        # Convert to PyTorch tensors
        for i, (hfilt, gfilt) in enumerate(zip(hfilters, gfilters)):
            hfilters[i] = torch.Tensor(hfilt).flatten()
            gfilters[i] = torch.Tensor(gfilt).flatten()

    # ... or initialize randomly.
    else:
        warnings.warn("No wavelet provided. CMFs will be initialized randomly.")
        hfilters = []
        for _ in range(n_filters):
            hfilters.append(_init_hfilter(filter_size))

        gfilters = None

    return hfilters, gfilters


def _init_hfilter(hfilter_size):
    """
    Random initialization of a low-pass filter (it might not generate an
    orthogonal wavelet basis).

    """
    hfilter = torch.rand(hfilter_size)
    reg = torch.sum(hfilter) / math.sqrt(2)
    hfilter /= reg # The sum of coefficients must be equal to sqrt(2)

    return hfilter


def _check_consistency_and_get_filtersize(
        list_of_hfilters, list_of_gfilters, n_filters
):
    assert len(list_of_hfilters) == n_filters
    if list_of_gfilters is not None:
        assert len(list_of_gfilters) == len(list_of_hfilters)
    filter_size = 0
    for i, hfilt in enumerate(list_of_hfilters):
        filter_size = max(filter_size, len(hfilt))
        if list_of_gfilters is not None:
            assert len(list_of_gfilters[i]) == len(hfilt)

    return filter_size


def _get_gfilter(hfilter):
    # ONLY WORKS WITH ORTHOGONAL FILTERS
    gfilter = torch.flip(torch.clone(hfilter), dims=(0,))
    gfilter[::2] *= -1 # Change one value out of 2 to its opposite
    return gfilter


def _duplicate_cmftypes(cmftypes, dupl, group=1):

    n_elts = len(cmftypes)
    out = []
    idx = 0
    while idx < n_elts:
        out += dupl * cmftypes[idx:idx+group]
        idx += group

    return out


def _get_next_cmftypes(cmftypes, cmf_map):
    out = []
    for cmftype in cmftypes:
        out += cmf_map[cmftype]
    return out


def _get_convkernels(cmftypes, cmf_dict):
    out = []
    for cmftype in cmftypes:
        out.append(cmf_dict[cmftype])
    out = torch.stack(out)
    return out


def _print_cmftypes(depth, orientation, cmftypes):
    print("Depth = {}; list of {} CMFs = {}".format(depth, orientation, cmftypes))


def _del_elts(inp, which, overwhich):
    """
    Delete elements from a list in a periodic way.

    Parameters
    ----------
    inp (list)
        Input list from which elements will be removed.
    which (list)
        List of indices.
    overwhich (int)
        Remove elements every xx indices, xx being given by this parameter.

    """
    which.sort(reverse=True) # Start with the last index
    out = inp.copy()
    for idx in which:
        del out[idx::overwhich]
        overwhich -= 1

    return out


def _del_featmaps(inp, which, overwhich):
    """
    Delete feature maps in a tensor in a periodic way.

    Parameters
    ----------
    inp (torch.tensor)
        Minibatch of input feature maps, of shape (n_imgs, n_channels, M, N).
    which (list)
        List of indices to delete in the channelwise dimension.
    overwhich (int)
        Remove feature maps every xx indices, xx being given by this parameter.

    """
    kept_channels = list(range(inp.shape[1]))
    kept_channels = _del_elts(kept_channels, which, overwhich)
    out = inp[:, kept_channels]

    return out


def _get_next_scale_bounds(scale_bounds, discard_featmaps):
    # Variable scale_bounds assumed to be sorted by descending order
    scale_bounds = [4 * bound for bound in scale_bounds]
    scale_bounds.append(1) # New scale (low-pass filter)
    discard_featmaps.sort(reverse=True) # Start with the last index
    for idx in discard_featmaps:
        for i, bound in enumerate(scale_bounds):
            if bound <= idx: # The current bound and all subsequents are not
                             # affected by the removal of the feature map
                break
            scale_bounds[i] -= 1 # One less feature map
    return scale_bounds
