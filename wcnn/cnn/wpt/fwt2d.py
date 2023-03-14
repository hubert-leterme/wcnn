from ._base_dwt2d import BaseDwt2d

CMF_MAP = dict(h=['h', 'g'], g=['h', 'g'])
INIT_HCMFTYPES = ['h']
INIT_GCMFTYPES = ['g']

class Wpt2d(BaseDwt2d):
    """
    Wavelet packet decomposition in a pseudo-local cosine basis as defined in
    [Mallat, 2009]. The corresponding admissible binary tree is a full tree whose
    depth is given as constructor parameter.

    Parameters
    ----------
    in_channels (int)
        Number of input channels.
    depth (int)
        Number of decomposition levels.
    wavelet (str, default=None)
        See dtcwt.coeffs.qshift for more details.
    backend (str, default='dtcwt')
        One of 'pywt' or 'dtcwt'. Name of the library from which to get the
        filters.
    trainable_qmf (bool, default=False)
        Whether to set the QMFs as trainable parameters.
    padding_mode (str, default='zeros')
        One of 'zeros' or 'reflect' ('periodic' may be implemented as well).

    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, cmf_map=CMF_MAP, init_hcmftypes=INIT_HCMFTYPES,
            init_gcmftypes=INIT_GCMFTYPES, **kwargs
        )

    @property
    def cmf_dict(self):
        return dict(h=self.hfilters[0], g=self.gfilters[0])

    @property
    def n_units(self):
        return 1
