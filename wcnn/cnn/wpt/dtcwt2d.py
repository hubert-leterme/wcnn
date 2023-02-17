import torch

from .. import cnn_toolbox
from ._base_dwt2d import BaseDwt2d

# CMF nomenclature
# ----------------
# h1fs, g1fs: primal low- and high-pass filters for the first stage
# h2fs, g2fs: dual low- and high-pass filters for the first stage (g1fs and g2fs
#             are one-sampled-shifted)
# h1, g1: primal low- and high-pass filters for the main stages
# h2, g2: dual low- and high-pass filters for the main stages (g1 and g2 are
#         approximately half-sample-shifted)
# h, g: low- and high-pass filters to apply on high-frequency coefficients
#       (primal and dual filters are identical)
CMF_MAP = dict(
    h1fs=['h1', 'g1'], g1fs=['h2', 'g2'],
    h2fs=['h2', 'g2'], g2fs=['h1', 'g1'],
    h1=['h1', 'g1'], g1=['h', 'g'],
    h2=['h2', 'g2'], g2=['h', 'g'],
    h=['h', 'g'], g=['h', 'g'],
)
INIT_HCMFTYPES = ['h1fs', 'h2fs']
INIT_GCMFTYPES = ['g1fs', 'g2fs']

# For a given number of decomposition stages, list of booleans indicating
# whether the support of the Fourier transform of each dual-tree filter is
# included in the half-plane of negative x-values.
NEG_FOURIER_SUPPORT = {
    2: torch.tensor([
        True, False,  True, False,  True, False,  True, False,  True,
       False,  True, False,  True, False,  True, False, False,  True,
       False,  True, False,  True, False,  True, False,  True, False,
        True, False,  True, False,  True
    ]),
    3: torch.tensor([
        True, False,  True, False, False, False, False, False,  True,
       False,  True, False, False, False, False, False,  True, False,
        True, False, False, False, False, False,  True, False,  True,
       False, False, False, False, False,  True, False,  True, False,
       False, False, False, False,  True, False,  True, False, False,
       False, False, False,  True, False,  True, False, False, False,
       False, False,  True, False,  True, False, False, False, False,
       False, False,  True, False,  True,  True,  True,  True,  True,
       False,  True, False,  True,  True,  True,  True,  True, False,
        True, False,  True,  True,  True,  True,  True, False,  True,
       False,  True,  True,  True,  True,  True, False,  True, False,
        True,  True,  True,  True,  True, False,  True, False,  True,
        True,  True,  True,  True, False,  True, False,  True,  True,
        True,  True,  True, False,  True, False,  True,  True,  True,
        True,  True
    ])
}

class DtCwpt2d(BaseDwt2d):
    """
    Module implementing the dual-tree complex wavelet packet transform.

    A complex wavelet is characterized by 2 1D low-pass filters ha and hb, from
    which we can build 4 2D low-pass filters: ha.ha, ha.hb, hb.ha and hb.hb.
    Since PyTorch computes cross-correlation instead of convolution, the so-called
    "convolution kernels" are actually cross-correlation kernels. The corresponding
    convolution filters (flipped arrays) are denoted bha and bhb. In order to
    yield nearly-analytic wavelets, bhb must be approximately half-sampled-delayed
    with respect to bha:
        BHb(omega) = exp(-i omega/2) BHa(omega)
    BHa and BHb being the Fourier transform of bha and bhb, respectively.
    The corresponding high-pass filters ga and gb can be deduced from ha and hb.

    Parameters
    ----------
    in_channels (int)
        Number of input channels.
    depth (int)
        Number of decomposition stages.
    stride (int)
        Subsampling level of the full decomposition.
    make_analytic (bool, default=True)
        If True, then feature maps whose Fourier support is included in the
        half-plane of negative x-values are replaced by their complex conjugates.
    split_featmaps (list of int, default=None)
    padding_mode (str, default='zeros')
        One of 'zeros' or 'reflect' ('periodic' may be implemented as well).
    wavelet (str, default=None)
        See dtcwt.coeffs.qshift for more details.
    backend (str, default='dtcwt')
        One of 'pywt' or 'dtcwt'. Name of the library from which to get the
        filters.

    Bibliography
    ------------
    Bayram, Ilker, and Ivan W. Selesnick. “On the Dual-Tree Complex
    Wavelet Packet and M-Band Transforms.” IEEE Transactions on Signal
    Processing, 2008. https://doi.org/10.1109/TSP.2007.916129.

    Selesnick, Ivan W., Richard Baraniuk, and Nick Kingsbury. “The
    Dual-Tree Complex Wavelet Transform.” IEEE Signal Processing Magazine 22,
    no. 6 (November 2005): 123–51. https://doi.org/10.1109/MSP.2005.1550194.

    """
    def __init__(
            self, in_channels, depth, stride, make_analytic=True, **kwargs
    ):
        self.make_analytic = make_analytic

        super().__init__(
            in_channels, depth, stride, cmf_map=CMF_MAP,
            init_hcmftypes=INIT_HCMFTYPES, init_gcmftypes=INIT_GCMFTYPES,
            n_filters=2, **kwargs
        )


    @property
    def hfilters_fs(self):
        """
        Low-pass filters for the first stage.

        """
        hfilt = self.hfilters[0]
        device = hfilt.device

        new_size = len(hfilt) + 2
        ha = torch.zeros(new_size, device=device)
        hb = torch.zeros(new_size, device=device)
        ha[2:] = hfilt
        hb[1:-1] = hfilt

        return [ha, hb]

    @property
    def gfilters_fs(self):
        """
        High-pass one-sampled-shifted filters for the first stage.

        """
        gfilt = self.gfilters[0]
        device = gfilt.device

        new_size = len(gfilt) + 2
        ga = torch.zeros(new_size, device=device)
        gb = torch.zeros(new_size, device=device)
        ga[:-2] = gfilt
        gb[1:-1] = -gfilt

        return [ga, gb]

    @property
    def cmf_dict(self):
        h1fs, h2fs = self.hfilters_fs
        g1fs, g2fs = self.gfilters_fs
        h1, h2 = self.hfilters
        g1, g2 = self.gfilters
        out = dict(
            h1fs=h1fs, g1fs=g1fs,
            h2fs=h2fs, g2fs=g2fs,
            h1=h1, g1=g1,
            h2=h2, g2=g2,
            h=h1, g=g1
        )
        return out

    @property
    def n_units(self):
        return 2


    def forward(self, x):
        """
        Computes the dual-tree complex wavelet transform.

        Parameter
        ---------
        x (torch.Tensor of shape (n_imgs, in_channels, M, N))
            Input batch of images.

        Returns
        -------
        z (torch.Tensor)
            Tensor of shape (n_imgs, out_channels, N//s, M//s, where s is the
            stride (subsampling factor).

        Let d denote the number of decomposition stages and psi^(i) denote the
        1D complex wavelet packets, for i = 0..(2**d - 1). Let's also denote
        bpsi^(i) their complex conjugate. Then we consider the following 2D
        wavelet packets:
            Psi_a^(ij) = psi^(i) x psi^(j);
            Psi_b^(ij) = psi^(i) x bpsi^(j),
        where 'x' denotes the tensor product. Let ya^(ij) and yb^(ij) denote the
        wavelet packet coefficients associated with Psi_a^(ij) and Psi_b^(ij),
        respectively. Then, for each image k, the output follows the following
        pattern (example with d = 2):

        z[k] = [
            ------- First input channel >> 64 first output channels -------

                ----- Coefficients related to Psi_a -----

                --- Top-left corner ---
                Re[ya^(00)], Im[ya^(00)],
                Re[ya^(01)], Im[ya^(01)],
                Re[ya^(10)], Im[ya^(10)],
                Re[ya^(11)], Im[ya^(11)],

                --- Top-right corner ---
                ...

                --- Bottom-left corner ---
                ...

                --- Bottom-right corner ---
                Re[ya^(22)], Im[ya^(22)],
                Re[ya^(23)], Im[ya^(23)],
                Re[ya^(32)], Im[ya^(32)],
                Re[ya^(33)], Im[ya^(33)],

                ----- Coefficients related to Psi_b -----

                --- Top-left corner ---
                Re[yb^(00)], Im[yb^(00)],
                Re[yb^(01)], Im[yb^(01)],
                Re[yb^(10)], Im[yb^(10)],
                Re[yb^(11)], Im[yb^(11)],

                ...

            ------- Second input channel >> 64 next output channels -------

            ...]

        The feature maps of complex coefficients ya^(ij) and yb^(ij) are computed
        by combining the four feature maps of real wavelet packet coefficients
        d_aa^(ij), d_ab^(ij), d_ba^(ij) and d_bb^(ij), computed during the dual-
        tree decomposition:
        ya^(ij) = [d_aa^(ij) - d_bb^(ij)] + i * [d_ba^(ij) + d_ab^(ij)]
        yb^(ij) = [d_aa^(ij) + d_bb^(ij)] + i * [d_ba^(ij) - d_ab^(ij)]

        """
        assert x.shape[1] == self._in_channels

        # Four wavelet packet decompositions
        x = super().forward(x)

        # Compute dual-tree coefficients ya^(ij) and yb^(ij)
        x = self._combine_dualtree_coefs(x)

        # Replace some feature maps by their complex conjugate, if required
        if self.make_analytic:
            x = self._get_complexconjugates(x)

        return x


    def get_resulting_kerstride(self):

        list_of_kernels = []
        list_of_strides = []

        # Four parallel WPT decompositions
        ker, stride = super().get_resulting_kerstride()
        list_of_kernels.append(ker)
        list_of_strides.append(stride)

        # Combine dual-tree feature maps
        list_of_kernels.append(cnn_toolbox.get_equivalent_convkernel(
            self._combine_dualtree_coefs, ker.shape[0]
        ))
        list_of_strides.append(1)

        # Replace some feature maps by their complex conjugate, if required
        if self.make_analytic:
            list_of_kernels.append(cnn_toolbox.get_equivalent_convkernel(
                self._get_complexconjugates, ker.shape[0]
            ))
            list_of_strides.append(1)

        ker, stride = cnn_toolbox.convolve_kernels(
            self._in_channels, list_of_kernels, list_of_strides
        )
        return ker, stride


    def _combine_dualtree_coefs(self, x):

        n_imgs, n_channels, height, width = x.shape

        # Split the second dimension into four groups (one for each filter bank)
        x = x.reshape(n_imgs, 4, -1, height, width)

        # Initialize output
        out = torch.zeros(
            n_imgs, 2, n_channels//2, height, width, device=x.device
        )

        # Coefficients related to Psi_a
        out[:, 0, 0::2] = x[:, 0] - x[:, 3] # Real part (even channels)
        out[:, 0, 1::2] = x[:, 2] + x[:, 1] # Imaginary part (odd channels)

        # Coefficients related to Psi_b
        out[:, 1, 0::2] = x[:, 0] + x[:, 3] # Real part (even channels)
        out[:, 1, 1::2] = x[:, 2] - x[:, 1] # Imaginary part (odd channels)

        # Group the feature maps by:
        # 1. input channel (e.g. 3 blocks of 64 feature maps if in_channels = 3)
        # 2. type of wavelet (one sub-group for Psi_a and one for Psi_b)
        out = out.view(n_imgs, 2, self._in_channels, -1, height, width)
        out = out.permute(0, 2, 1, 3, 4, 5).clone() # Clone is necessary to avoid
                                                    # UnsafeViewBackward while
                                                    # reshaping tensor (very slow
                                                    # in the backprop phase).
        # Merge dimensions
        out = out.reshape(n_imgs, -1, height, width)

        return out


    def _get_complexconjugates(self, x):
        """
        Replace the imaginary part of some feature maps by their opposite.

        """
        if self.depth not in (2, 3):
            raise NotImplementedError

        n_imgs, out_channels, height, width = x.shape
        x = x.view(n_imgs, self._in_channels, -1, 2, height, width)
        x0 = x.clone()
        x[
            :, :, NEG_FOURIER_SUPPORT[self.depth], 1
        ] = -x0[
            :, :, NEG_FOURIER_SUPPORT[self.depth], 1
        ] # Avoids in-place operation
        x = x.reshape(n_imgs, out_channels, height, width)

        return x
