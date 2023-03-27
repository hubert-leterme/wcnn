"""
The functions defined here are only intended for the 'plots' module as well as
external usage. Any function that may be imported by other modules must be
placed in the 'toolbox' module. This is done to prevent circular imports.

"""
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cnn import cnn_toolbox, wpt

SPECTR_SIZE = 65

def get_scores(score_dict):
    """
    Arguments:
    ----------
    score_dict (dict): dictionary of scores (keys: epoch number; values: score or
        dictionary of scores in case of multiple metrics)

    Returns:
    --------
    epoch (list): list of epochs
    score (list): list of scores. In case of multiple metrics, only the top-1
        accuracy score is kept.

    """
    epoch = []
    score = []
    for ep in sorted(score_dict):
        assert isinstance(ep, int)
        epoch.append(ep)
        sc = score_dict[ep]
        if isinstance(sc, float):
            score.append(sc)
        elif isinstance(sc, dict):
            assert isinstance(sc[1], float)
            score.append(sc[1])
        else:
            raise ValueError('Unrecognized format for a score: {}.'.format(
                type(sc)
            ))

    return epoch, score


def aggregate(x, aggr, action=np.average):

    x = np.array(x)

    # Truncate the list to be able to split into equal size arrays
    n0 = x.size
    n = n0 - (n0 % aggr)
    x = x[:n]

    # Split array
    n_elts = n // aggr
    x = np.split(x, n_elts)

    return [action(x0) for x0 in x]


def concat_sidebyside(x):
    """
    Group input channels four by four and concatenate them side by side.

    Parameter
    ---------
    x (torch.Tensor): tensor of shape (k, in_channels, m, n)

    Returns
    -------
    y (torch.Tensor): tensor of shape (k, in_channels/4, 2*m, 2*n)

    """
    cat1 = torch.cat((x[:, 0::4], x[:, 1::4]), dim=3)
    cat2 = torch.cat((x[:, 2::4], x[:, 3::4]), dim=3)

    return torch.cat((cat1, cat2), dim=2)


def recursive_concat_sidebyside(x, depth, return_abs=True,
                                set_topleft_tensor_to_zero=True):
    """
    Parameter
    ---------
    x (torch.Tensor): tensor of shape (k, in_channels, m, n)
    depth (int): number of recursions
    return_norm (bool): whether to return the absolute value of the tensors

    Returns
    -------
    y (torch.Tensor): tensor of shape (k, in_channels/(4**depth),
        (2**depth)*m, (2**depth)*n)

    """
    for _ in range(depth):
        x = concat_sidebyside(x)

    # Set top-left subtensor to 0 (scaling coefficients)
    if set_topleft_tensor_to_zero:
        m, n = x.shape[2:]
        x[:, :, :m//(2**depth), :n//(2**depth)] = 0

    if return_abs:
        x = torch.abs(x)

    return x


def crop_images(imgs, output_size=None, beg=None):

    norm_all = np.linalg.norm(imgs, axis=(-2, -1))

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    m, n = imgs.shape[-2:]
    if beg is None:
        beg = (max(m - output_size[0], 0)//2, max(n - output_size[1], 0)//2)
    elif isinstance(beg, int):
        beg = (beg, beg)
    imgs = imgs[
        ...,
        beg[0]:beg[0]+output_size[0],
        beg[1]:beg[1]+output_size[1]
    ]

    norm_crop = np.linalg.norm(imgs, axis=(-2, -1))

    return imgs, norm_all, norm_crop


def fft2(weight, **kwargs):
    """
    Parameters
    ----------
    weight (torch.tensor)
        Tensor of size (out_channels, in_channels, height, width)
    view_as_complex (bool or list of bool, default=False)
    groups (list of int, default=None)
        List of lengths of channel groups. Only useful if 'view_as_complex' is
        set to True, or to a list containing True. Must be of same length as
        'view_as_complex'.
    complex_input (bool, default=False)
    output_size (int, default=65)
    return_abs (bool, default=True)

    """
    return _fft2(weight, **kwargs)


def ifft2(weight, **kwargs):
    """
    Parameters
    ----------
    weight (torch.tensor)
        Tensor of size (out_channels, in_channels, height, width)
    view_as_complex (bool or list of bool, default=False)
    groups (list of int, default=None)
        List of lengths of channel groups. Only useful if 'view_as_complex' is
        set to True, or to a list containing True. Must be of same length as
        'view_as_complex'.
    complex_input (bool, default=False)
    output_size (int, default=65)
    return_abs (bool, default=True)

    """
    return _fft2(weight, inverse=True, **kwargs)


def _fft2(
        weight, view_as_complex=False, groups=None,
        output_size=SPECTR_SIZE, return_abs=True, inverse=False
):
    _, in_channels, height, width = weight.shape

    if groups is None:
        groups = [weight.shape[0]]
    if isinstance(view_as_complex, bool):
        view_as_complex = len(groups) * [view_as_complex]
    else:
        assert len(view_as_complex) == len(groups)
        assert sum(groups) == weight.shape[0]

    out = []
    inf_chan = 0
    for vac, grp in zip(view_as_complex, groups):
        sup_chan = inf_chan + grp
        wght = weight[inf_chan:sup_chan]
        if vac:
            # Shape (out_channels / 2, 2, in_channels, height, width)
            wght = wght.view(-1, 2, in_channels, height, width)

            # Shape (out_channels / 2, in_channels, width, height, 2)
            # Apply .contiguous() to avoid RuntimeError (see https://bit.ly/2WFn6DD)
            wght = wght.permute(0, 2, 3, 4, 1).contiguous()

            # Complex tensor, shape (out_channels / 2, in_channels, width, height)
            wght = torch.view_as_complex(wght)

        out.append(wght)
        inf_chan = sup_chan

    out = torch.cat(out, dim=0)

    if output_size is not None:
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

    if not inverse:
        out = torch.fft.fft2(out, s=output_size)
        out = torch.fft.fftshift(out, dim=(-2, -1))
    else:
        out = torch.fft.fftshift(out, dim=(-2, -1))
        out = torch.fft.ifft2(out, s=output_size)
    if return_abs:
        out = torch.abs(out)

    return out


def directional_centers_of_mass(spectr, prefered_direction, power_ponderation=2,
                                return_ponderations=False):

    theta = torch.Tensor([
        [torch.pi, 0, 0],
        [0, -torch.pi, 0]
    ]) # Frequencies belong to [-pi, pi] x [-pi, pi]
    n_imgs, in_channels, _, _ = spectr.shape
    theta = torch.stack(n_imgs * [theta])
    grid = F.affine_grid(theta, spectr.shape)
    grid = torch.stack(in_channels * [grid], dim=1)
    grid = grid.permute((-1, 0, 1, 2, 3))
    prefered_direction = prefered_direction.permute((-1, 0, 1))

    innerprods = _inner_products(grid, prefered_direction)
    ponderations = spectr**power_ponderation
    ponderations = innerprods**2 * ponderations
    sum_ponderations = torch.sum(ponderations, dim=(-2, -1))

    out = torch.sum(ponderations * grid, dim=(-2, -1)) / sum_ponderations

    # Return only vectors with positive x-component. If this composant is equal
    # to 0, then the y-component must be positive.
    sign_x = torch.sign(out[0, :, :])
    sign_y = torch.sign(out[1, :, :])
    sign = torch.where(torch.abs(sign_x) > 0.5, sign_x, sign_y)
    out *= sign

    out = out.permute((1, 2, 0))

    if return_ponderations:
        out = (out, ponderations)

    return out


def _inner_products(grid, prefered_direction):

    grid_norm = torch.sqrt(grid[0]**2 + grid[1]**2)
    grid_directions = grid / grid_norm # Normalized affine grid
    # Replace all nan values by 1 (should only be at the origin)
    grid_directions[grid_directions != grid_directions] = 1.

    # Get a grid with repeated prefered direction (for inner product)
    m, n = grid_directions.shape[-2:]
    prefered_direction = torch.stack(m * [prefered_direction], dim=-1)
    prefered_direction = torch.stack(n * [prefered_direction], dim=-1)

    out = F.relu(torch.sum(prefered_direction * grid_directions, dim=0))

    return out


def kernel_characterization(inp, spectr_size=SPECTR_SIZE, rgb=True,
                            return_dataframe=True, **kwargs):
    """
    The set of kernels connecting each input feature map to a given output is
    characterized by a N-dimensional vector, with N = 5K + (K-1), K being the
    number of input channels. More precisely, each convolution kernel is
    characterized by a 5-dimensional vector, representing:
    - normalized sum of its values;
    - main frequency (2D vector);
    - homogeneity;
    - coherence index.
    In addition, the norm-1 ratios with respect to the first input kernel are
    computed.

    Parameters
    ----------
    inp (torch.Tensor of shape (out_channels, in_channels, n, n))
        The convolution kernels of size (n, n)
    spectr_size (int, default=SPECTR_SIZE)
        Output size of the fast Fourier transform
    rgb (bool, default=True)
        If True, then the elementwise sum of convolution kernels is performed
        over the input channels before computing the kernel characteristics.
        This is equivalent to consider a convolution layer taking as input the
        linear projection of RGB images into a color space (e.g., YCbCr). The
        characteristics are then computed on the convolution kernel applied on
        the luminance component of images.
    return_dataframe (bool, default=False)
        If True, return a Pandas dataframe instead of a PyTorch tensor.

    Returns
    -------
    Either:
    - out (torch.Tensor of shape (out_channels, N) or pandas.DataFrame with
           number of rows equal to out_channels and N columns).
        N is equal to
            - 5K + (K-1) if rgb is False;
            - 5 + (K-1) if rgb is True,
        where K is equal to in_channels.

    """
    # Compute the elementwise sum of RGB filters, if 'rgb' is set to True
    if rgb:
        inp = torch.sum(inp, dim=1).unsqueeze(dim=1)

    out_channels, in_channels = inp.shape[:2]

    # Normalize input according to norm-1 (separately for each feature map)
    inp = inp.permute((-2, -1, 0, 1))
    norms = torch.norm(inp, p=1, dim=(0, 1))
    inp_normalized = inp / norms

    inp = inp.permute((-2, -1, 0, 1))
    inp_normalized = inp_normalized.permute((2, 3, 0, 1))

    struct = cnn_toolbox.structure_tensor(inp_normalized)
    homog, arrow, coher = cnn_toolbox.homogeneity_direction_coherence(struct)
    spectr = fft2(inp_normalized, output_size=spectr_size)
    nu = directional_centers_of_mass(spectr, arrow, **kwargs)
    sumvals = torch.abs(torch.sum(inp_normalized, dim=(-2, -1)))

    sumvals = torch.unsqueeze(sumvals, -1)
    homog = torch.unsqueeze(homog, -1)
    coher = torch.unsqueeze(coher, -1)

    out = torch.cat([sumvals, nu, homog, coher], dim=-1)

    # Shape = (out_channels, 5 * in_channels)
    out = out.reshape(out.shape[0], -1)

    if return_dataframe:
        columns_df = ['norm_sum', 'freq_x', 'freq_y', 'homogeneity', 'coherence']
        columns_df = in_channels * columns_df
        index = pd.Index(np.arange(out_channels), name='feature_map')
        out = pd.DataFrame(out.numpy(), columns=columns_df, index=index)
        out['freq_norm2'] = np.sqrt(out['freq_x']**2 + out['freq_y']**2)
        out['freq_norminfty'] = np.maximum(
            np.abs(out['freq_x']), np.abs(out['freq_y'])
        )

    return out


def get_dtcwpt_filter_idx(depth):
    r"""
    Indicate how to plot the values in the frequency plane.

    """
    dtcwpt = wpt.DtCwpt2d(
        in_channels=1, depth=depth, stride=2**depth
    )
    ker = dtcwpt.resulting_kernel
    charact = kernel_characterization(
        ker[::2], rgb=False, power_ponderation=4, return_dataframe=True
    )
    charact = charact[['freq_x', 'freq_y']]

    # Symmetrize filters
    index_conj = get_index_conj(depth)
    charact_conj = pd.DataFrame(
        data=-charact.to_numpy(), columns=['freq_x', 'freq_y'], index=index_conj
    )
    charact = charact.append(charact_conj)
    out = charact // (np.pi / (2**depth))
    out = out.astype(int)

    return out


def convert_to_ycckernel(weight, alphab, alphar):
    """
    Given a weight tensor trained on RGB images, return the equivalent weight
    tensor if the CNN was trained on YCbCr images.

    """
    transinv_colmat = get_transinvcolormatrix(alphab, alphar)
    weight = weight.permute(2, 3, 1, 0)
    weight_ycc = torch.matmul(transinv_colmat, weight).permute(3, 2, 0, 1)

    return weight_ycc


def get_colormatrix(alphab, alphar):
    """
    Get a color matrix.
    See https://en.wikipedia.org/wiki/YCbCr for more details.

    """
    alphag = _check_alpha(alphab, alphar)

    out = torch.tensor([
        [alphar, alphag, alphab],
        [-0.5 * alphar/(1-alphab), -0.5 * alphag/(1-alphab), 0.5],
        [0.5, -0.5 * alphag/(1-alphar), -0.5 * alphab/(1-alphar)]
    ])
    return out


def get_transinvcolormatrix(alphab, alphar):
    """
    Get the tranposed inverse of a color matrix.
    See https://en.wikipedia.org/wiki/YCbCr for more details.

    """
    alphag = _check_alpha(alphab, alphar)

    valb = 2 - 2 * alphab
    valr = 2 - 2 * alphar

    out = torch.tensor([
        [1., 1., 1.],
        [0., -alphab/alphag * valb, valb],
        [valr, -alphar/alphag * valr, 0.]
    ])
    return out


def _check_alpha(alphab, alphar):

    alphag = 1 - alphab - alphar
    assert alphar > 0 and alphar < 1
    assert alphag > 0 and alphag < 1
    assert alphab > 0 and alphab < 1

    return alphag


def tolist(var, n):

    if not isinstance(var, list):
        var = n * [var]
    else:
        assert len(var) == n

    return var


def get_meangrads(classifs, list_of_out_channels, j, idx=None):
    """
    Return the mean gradients over all the output channels of a given scale.

    Parameters
    ----------
    classifs (list of wcnn.classifier.WaveCnnClassifier)
        List of trained classifiers. Attribute 'grads_' must be registered.
    list_of_out_channels (list of int)
        Number of output channels for each DT-CWPT scale > 0.
    j (int > 0)
        Scale for which to compute the gradients.
    idx (list of int, default=None)
        If provided, list of indices to select a bunch of classifiers among
        the provided list.

    """
    assert j > 0
    out = []
    if idx is None:
        idx = range(len(classifs))
    for i in idx:
        grd = classifs[i].grads_[j-1]
        grd = grd.view(-1, list_of_out_channels[j-1])
        grd_avg = torch.mean(grd, dim=1)
        out.append(grd_avg)

    return out


def get_index_conj(
        depth, feature_map=None, extra_cols=None, extra_cols_names=None
):
    n_filters = _get_n_filters(depth)

    if feature_map is None:
        feature_map = range(n_filters, 2 * n_filters)
    else:
        feature_map = [k + n_filters for k in feature_map]

    if extra_cols is None:
        out = pd.Index(data=feature_map, name='feature_map')
    else:
        out = pd.MultiIndex.from_product(
            [feature_map, *extra_cols],
            names=['feature_map', *extra_cols_names]
        )

    return out


def _get_n_filters(depth):
    return 2 * 4**depth


def assess_gaborfilters(weight, m, fftsize):
    """
    Compute the maximum percentage of energy of a set of convolution kernels
    within a squared Fourier window of size (kappa x kappa), where the bandwidth
    kappa is chosen to be equal to (pi / m). 

    Parameters
    ----------
    weight (torch.Tensor)
        Tensor of shape (out_channels, in_channels, height, width).
    m (int)
        Value characterizing the Fourier window size (kappa := pi / m). We recommend to use
        the subsampling factor (stride) of the convolution layer.
    fftsize (int)
        Output size when computing the 2D FFT. The results may be more precise if
        fftsize is large.

    Returns
    -------
    rho (torch.Tensor)
        Tensor of shape (out_channels,), containing the maximum percentage of energy
        for each output channel.
    characfreqs (torch.Tensor)
        Tensor of shape (out_channels, 2), containing the characteristic frequencies
        for each output channel.
    
    """
    assert fftsize % (2 * m) == 0
    kappa = fftsize // (2 * m)
    avgwindow = torch.ones((1, 1, kappa, kappa))
    padding = (kappa - 1) // 2

    # Compute the mean kernel over the input channels
    # Shape = (out_channels, 1, height, width)
    avgweight = torch.mean(weight, dim=1).unsqueeze(1)

    # Compute the FFT of the analytic kernels W := V + i H(V)
    complweight_fft = cnn_toolbox.get_analytic_kernel(
        avgweight, s=(fftsize, fftsize), fft=True
    )

    # Compute the power spectrum
    powerspectr_fft = torch.abs(complweight_fft)**2
    powerspectr_fft_pad = F.pad(
        powerspectr_fft, (padding, padding + 1, padding, padding + 1), mode='circular'
    )

    # Compute the kernel's energy within a sliding Fourier window of size (kappa x kappa)
    powerspectr_fft_window = F.conv2d(powerspectr_fft_pad, weight=avgwindow).squeeze(1)
    powerspectr_fft_window_0 = powerspectr_fft_window.flatten(start_dim=1)

    # Get the maximum energy within such a Fourier window, as well as the corresponding
    # position in the Fourier domain (characteristic frequency)
    maxenergy_fft_window, idx_characfreqs = torch.max(powerspectr_fft_window_0, dim=-1)
    freqs = cnn_toolbox.get_freqs((fftsize, fftsize)) # Generate a grid of frequency coordinates
    characfreqs = 2 * torch.pi * freqs.flatten(end_dim=1)[idx_characfreqs]

    # Normalize the maximum energy with respect to the kernel's total energy
    maxenergy_fft = torch.sum(powerspectr_fft.squeeze(1), dim=(-2, -1))
    rho = maxenergy_fft_window / maxenergy_fft

    return rho, characfreqs
