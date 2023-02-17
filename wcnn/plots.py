import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as col
import torch

from . import plots_toolbox, toolbox
from .cnn import wpt, cnn_toolbox

CMAP = 'PiYG'
FONTSIZE_TITLE = 22
FIGSIZE = (5, 5)

LIST_OF_KWARGS_LEGEND = ["legend_loc", "ncol", "labelspacing"]

class MidpointNormalize(col.Normalize):
    """
    Normalise the colorbar.

    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):

        xp = [self.vmin, self.vmax]
        vdiff_upper = self.vmax - self.midpoint
        vdiff_lower = self.midpoint - self.vmin
        if vdiff_upper == 0 and vdiff_lower == 0:
            fpmin = -1
            fpmax = 1
        elif vdiff_upper >= vdiff_lower:
            fpmax = 1
            fpmin = 1 - (self.vmax - self.vmin) / (2 * vdiff_upper)
        else:
            fpmin = 0
            fpmax = (self.vmax - self.vmin) / (2 * vdiff_lower)
        fp = [fpmin, fpmax]

        return np.ma.masked_array(np.interp(value, xp, fp), np.isnan(value))


class PolynomialNorm(col.Normalize):

    def __init__(self, degree=1., vmin=None, vmax=None, clip=False):

        self.degree = degree
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):

        scale = self.vmax - self.vmin
        shift = self.vmin / (self.vmax - self.vmin)

        out = (value / scale - shift)**self.degree

        return np.ma.masked_array(out, np.isnan(value))


def plot_data(
        list_of_x, list_of_dataseqs, labels, aggr,
        alpha=None, color=None, markers=None, markersizes=None, linestyles=None,
        figsize=FIGSIZE, min_x=None, max_x=None, xlabel=None, ylabel=None,
        yscale="linear", xleft=None, xright=None, ybottom=None, ytop=None,
        title=None, markevery=1, zorders=None, select_idx=None,
        fontsize_title=FONTSIZE_TITLE, **kwargs
):
    kwargs_legend = toolbox.extract_dict(LIST_OF_KWARGS_LEGEND, kwargs)

    n = len(list_of_x)
    assert len(list_of_dataseqs) == n
    assert len(labels) == n
    if select_idx is None:
        select_idx = range(n)

    # Aggregate data if needed
    list_of_dataseqs_aggr = []
    list_of_x_aggr = []
    for x, seq in zip(list_of_x, list_of_dataseqs):
        if seq is not None:
            list_of_dataseqs_aggr.append(plots_toolbox.aggregate(seq, aggr))
            list_of_x_aggr.append(x[::aggr])
        else:
            list_of_dataseqs_aggr.append(None)
            list_of_x_aggr.append(None)

    # Initialize key word arguments
    list_of_kwargs = _init_kwargs(
        n, alpha=alpha, color=color, markers=markers, markersizes=markersizes,
        linestyles=linestyles, zorders=zorders, markevery=markevery, **kwargs
    )

    # Figure
    plt.figure(figsize=figsize)

    # Plot data
    out = []
    for i, (x, seq, kwargs) in enumerate(zip(
            list_of_x_aggr, list_of_dataseqs_aggr, list_of_kwargs
    )):
        if i in select_idx and seq is not None:
            if isinstance(x, torch.Tensor):
                x = x.detach().numpy()
            elif isinstance(x, list):
                x = np.array(x)
            if isinstance(seq, torch.Tensor):
                seq = seq.detach().numpy()
            elif isinstance(seq, list):
                seq = np.array(seq)
            min_x0 = min_x if min_x is not None else min(x)
            max_x0 = max_x if max_x is not None else max(x)
            x = x[:len(seq)]
            x0 = x[(x >= min_x0) & (x <= max_x0)]
            seq = seq[(x >= min_x0) & (x <= max_x0)]

            out.append(plt.plot(x0, seq, **kwargs))

    # Chart layout
    plt.legend([labels[i] for i in select_idx], **kwargs_legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    if xleft is not None:
        plt.xlim(left=xleft)
    if xright is not None:
        plt.xlim(right=xright)
    if ybottom is not None:
        plt.ylim(bottom=ybottom)
    if ytop is not None:
        plt.ylim(top=ytop)
    plt.title(title, size=fontsize_title)

    return out


def plot_loss(
        list_of_classifs, aggr=1,
        xlabel="Number of iterations", ylabel="Cross-entropy loss",
        **kwargs
):
    out = _plot_data_from_classif(
        list_of_classifs, 'losses_', aggr=aggr, xlabel=xlabel, ylabel=ylabel,
        **kwargs
    )
    return out


def plot_validation_errs(
        list_of_classifs, xlabel="Number of epochs",
        ylabel="Top-1 validation error", aggr=1, **kwargs
):
    out = _plot_data_from_classif(
        list_of_classifs, 'val_errs_', aggr=aggr, xlabel=xlabel, ylabel=ylabel,
        transf=plots_toolbox.get_scores, **kwargs
    )
    return out

EPOCH_RELATED_ATTRIBUTES = ['val_errs_', 'val_times_']

def _plot_data_from_classif(list_of_classifs, attr_name, aggr, labels=None,
                            transf=None, markevery=5, **kwargs):

    n = len(list_of_classifs)
    if transf is not None:
        list_of_x = []
    else:
        list_of_x = None
    list_of_dataseqs = []
    list_of_names = []
    for classif in list_of_classifs:
        seq = getattr(classif, attr_name)
        if transf is not None:
            x, seq = transf(seq)
            # Remove datapoints if required
            if attr_name in EPOCH_RELATED_ATTRIBUTES:
                for epoch in classif.remove_epochs_:
                    ind = x.index(epoch)
                    x.pop(ind)
                    seq.pop(ind)
            list_of_x.append(x)
        list_of_dataseqs.append(seq)
        list_of_names.append(classif.name)

    if labels is None:
        labels = list_of_names

    list_of_dataseqs = list_of_dataseqs.copy()

    # Compute maximum number of datapoints
    if list_of_x is None:
        n_datapoints = 0
        for seq in list_of_dataseqs:
            k = len(seq)
            if k > n_datapoints:
                n_datapoints = k
        x = np.arange(n_datapoints) + 1 # x axis
        list_of_x = n * [x]

    return plot_data(
        list_of_x, list_of_dataseqs, labels, aggr, markevery=markevery, **kwargs
    )


def visualize_imgs(imgs, rgb=True, channel=None, output_size=None, beg=None,
                   figsize=FIGSIZE, nrows=None, normalize=False,
                   zero_centered=True, saturation=1., alpha=0.2,
                   selected_idx=None, **kwargs):
    """
    Plot mosaic of images.

    Parameters
    ----------
    imgs (torch.tensor or numpy.ndarray)
        Array of shape (n_imgs, in_channels, N, N), where N is the size of the
        image.
    rgb (bool, default=True)
        Whether to plot gray-scale or color images.
    channel (int)
        If rgb=False, input channel from which to visualize the images.
    output_size (int or tuple of int, default=None)
        Crop images to the desired size.
    beg (int or tuple of int, default=None)
        Pixel from which to start cropping.
    figsize (int or tuple of int, default=10)
    nrows (int, default=None)
    normalize (bool, default=False)
        If True, normalize images over all the input channels. This flag should
        be used if the input has negative values.
    zero_centered (bool, default=True)
        If set to True, then 0 is considered at the center value while normalizing.
        This flag should be used if the input has negative values.
    saturation (float, default=1.)
        Defines the normalization bounds as a percentage of the minimum and
        maximum values of each image.
    alpha (float or list of float, default=0.2)
    selected_idx (list of int, default=None)
        Specify the indices of the images to highlight. If specified, then 'alpha'
        will be discarded.
    **kwargs
        Extra arguments for matplotlib.pyplot.imshow

    """
    imgs, n_imgs, nrows, ncols = _prepare_images(
        imgs, output_size, nrows, beg=beg
    )

    if selected_idx is not None:
        alpha = 64 * [alpha]
        for l in selected_idx:
            alpha[l] = 1.
    else:
        alpha = plots_toolbox.tolist(alpha, n_imgs)

    plt.figure(figsize=figsize)
    for i in range(n_imgs):
        img = imgs[i]
        if normalize:
            img = _normalize(
                img, saturation=saturation, zero_centered=zero_centered
            )
            kwargs.update(vmin=0., vmax=1.)
        if not rgb:
            img = img[channel]
        else:
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2) # Shape(M, N, 3)

        if selected_idx is not None:
            kwargs.update(alpha=alpha[i])

        plt.subplot(nrows, ncols, i+1)
        out = plt.imshow(img, **kwargs)
        plt.axis('off')

    return out


def visualize_channels(inp, output_size=None, beg=None, figsize=(6, 6), nrows=None,
                       normalize=False, zero_centered=True, saturation=1.,
                       alpha=None, **kwargs):
    """
    Plot mosaic of images; each channel is considered as one image.

    Parameters
    ----------
    inp (torch.tensor or numpy.ndarray)
        Array of shape (n_channels, N, N), where N is the size of the images.
    output_size (int or tuple of int, default=None)
        Center crop images to the desired size.
    beg (int or tuple of int, default=None)
        Pixel from which to start cropping.
    figsize (int or tuple of int, default=10)
    nrows (int, default=None)
    normalize (bool, default=False)
        If True, normalize images over all the channels. This flag should be
        used if the input has negative values.
    zero_centered (bool, default=True)
        If set to True, then 0 is considered at the center value while normalizing.
        This flag should be used if the input has negative values.
    saturation (float, default=1.)
        Defines the normalization bounds as a percentage of the minimum and
        maximum values of each image.
    alpha (float or list of float, default=None)
    **kwargs
        Extra arguments for matplotlib.pyplot.imshow

    """
    inp, n_channels, nrows, ncols = _prepare_images(
        inp, output_size, nrows, beg=beg
    )
    if alpha is not None:
        alpha = plots_toolbox.tolist(alpha, n_channels)

    if normalize:
        inp = _normalize(
            inp, saturation=saturation, zero_centered=zero_centered
        )
        kwargs.update(vmin=0., vmax=1.)

    plt.figure(figsize=figsize)
    for i in range(n_channels):
        img = inp[i]
        if alpha is not None:
            kwargs.update(alpha=alpha[i])
        plt.subplot(nrows, ncols, i+1)
        out = plt.imshow(img, **kwargs)
        plt.xticks([])
        plt.yticks([])

    return out


def _prepare_images(imgs, output_size, nrows, **kwargs):

    # Crop images
    if output_size is None:
        output_size = tuple(imgs.shape[-2:])
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()
    imgs, _, _ = plots_toolbox.crop_images(
        imgs, output_size=output_size, **kwargs
    )

    # Get number of images, rows and columns
    n_imgs = imgs.shape[0]
    if nrows is None:
        nrows = -(math.floor(-math.sqrt(n_imgs)))
        ncols = nrows
    else:
        ncols = -(math.floor(-n_imgs // nrows))

    return imgs, n_imgs, nrows, ncols


def _normalize(inp, saturation, zero_centered):

    vmin = saturation * inp.min()
    vmax = saturation * inp.max()
    if zero_centered:
        # In this case, 0 set considered as the median value (gray).
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0.)
    else:
        # The input is considered as being a positive-valued array, with 0 set
        # to black and the maximum value set to white.
        norm = col.Normalize(vmin=0., vmax=vmax)
    inp = norm(inp)

    # Note: a simpler solution would have been to pass the normalizer
    # into the keyword arguments, but unfortunately this argument is
    # ignored for RGB data. See
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    # for more details.

    return inp


def visualize_convkernels(weight, cmap='bone', **kwargs):
    """
    Plot convolution kernels as a mosaic of images.

    Parameters
    ----------
    weight (torch.tensor or numpy.ndarray)
        Array of shape (out_channels, in_channels, n, n), where n is the size of
        the convolution kernels.
    channel (int, default=None)
        Input channel from which to visualize the convolution kernels.
        Actually each input channel gets its own convolution kernels, one for
        each output channel.
    rgb (bool, default=True)
        If True, then color filters are displayed.
    output_size (int or tuple of ints, default=None)
        If specified, crop weight to get the desired size.
    cmap (str, default='bone')
        Color map. Only if `rgb` is True.

    """
    if isinstance(weight, torch.Tensor):
        weight = weight.numpy()
    return visualize_imgs(weight, cmap=cmap, normalize=True, **kwargs)


def fft2(weight, cmap='bone', return_spectr=False, **kwargs):
    """
    Plot the Fourier transform of convolution kernels as a mosaic of images.

    Parameters
    ----------
    weight (torch.tensor or numpy.ndarray)
        Array of shape (out_channels, in_channels, n, n), where n is the size of
        the convolution kernels.
    channel (int, default=None)
        Input channel from which to visualize the convolution kernels.
        Actually each input channel gets its own convolution kernels, one for
        each output channel.
    rgb (bool, default=True)
        If True, then color filters are displayed.
    view_as_complex (bool or list of bool, default=False)
    output_size (int or tuple of ints, default=None)
        Output size or shape. If none is given, then the output has the same
        size as the input. See documentation of `numpy.fft.fft2` (parameter `s`)
        for more details.
    cmap (str, default='bone')
        Color map. Only if `rgb` is True.
    return_spectr (bool, default=False)
        If True, then returns the array of Fourier coefficients.

    """
    list_of_kwargs_fft2 = ['view_as_complex', 'groups', 'output_size']
    kwargs_fft2 = toolbox.extract_dict(list_of_kwargs_fft2, kwargs)
    spectr = plots_toolbox.fft2(weight, **kwargs_fft2)
    out = visualize_imgs(
        spectr, cmap=cmap, normalize=True, zero_centered=False, **kwargs
    ) # zero_centered = False because the arrays have only positive values
    if return_spectr:
        out = (out, spectr)
    return out


def visualize_wavecoefs(inp, transf, mode=None, **kwargs):
    """
    Parameters
    ----------
    inp (torch.Tensor)
        Tensor of shape (3, height, width).
    transf (wcnn.cnn.wpt._base_dwt2d.BaseDwt2d)
        PyTorch module representing a wavelet packet transform.
    mode (str, default=None)
        If transf is of type wcnn.cnn.wpt.DtCwpt2d, then this argument must
        be specified. Can be one of:
        - 'real': display the real part of complex coefficients;
        - 'modulus': display the modulus of complex coefficients.

    """
    # Compute the number of display units (1 for standard WPT and 2 for
    # dual-tree transforms)
    if isinstance(transf, wpt.Wpt2d):
        n_squares = 1
    elif isinstance(transf, wpt.DtCwpt2d):
        n_squares = 2
    else:
        raise TypeError
    in_channels = inp.shape[0]
    inp = torch.unsqueeze(inp, 0) # Embed a single image into a minibatch

    # Compute wavelet coefficients
    # Shape = (1, out_channels, M/stride, N/stride)
    out = transf(inp)
    if mode == 'real':
        out = cnn_toolbox.get_real_part(out)
    elif mode == 'modulus':
        out = cnn_toolbox.get_modulus(out)

    height, width = out.shape[-2:]

    # Shape = (in_channels, n_squares, K, M/stride, N/stride), where K denotes
    # the number of feature maps per display unit and per input channel
    out = out.reshape(in_channels, n_squares, -1, height, width)

    # Shape = (n_squares, in_channels, K, M/stride, N/stride)
    out = out.permute(1, 0, 2, 3, 4)

    # Shape = (1, out_channels, M/stride, N/stride) (output channels are
    # re-arranged)
    out = out.reshape(1, -1, height, width)
    height *= 2**transf.depth
    width *= 2**transf.depth
    out = plots_toolbox.recursive_concat_sidebyside(out, transf.depth).view(
        n_squares, in_channels, height, width
    )

    return visualize_imgs(out, **kwargs)


def scatter_centergrid(inps, color=None, markers=None, markersizes=None, figsize=FIGSIZE,
                       xmax=np.pi, ymax=np.pi, symmetric=True, labels=None,
                       title=None, xlabel=None, ylabel=None, fontsize_title=FONTSIZE_TITLE,
                       legend_loc=None, zorders=None, selection=None, **kwargs):
    """
    Parameters
    ----------
    inps (array-like or list of array-like, shape (n_points, 2))
        The (x, y) coordinates of every points to display.
    color (list of color arguments, default=None)
        See matplotlib.pyplot.scatter (c=) for more details.
    figsize (tuple of floats, optional)
    xmax, ymax (float, default=0.5)
    symmetric (bool, default=True)
        If True, plot points and their symmetric image.
    labels (list of str, default=None)
        Labels for each point cloud.
    **kwargs
        Keyword arguments for plt.scatter.

    """
    if not isinstance(inps, list):
        inps = [inps]
        color = [color]
        markers = [markers]
        labels = [labels]
    if color is None:
        color = len(inps) * [None]
    if markers is None:
        markers = len(inps) * [None]
    if labels is None:
        labels = len(inps) * [None]

    if symmetric:
        for i, c in enumerate(color):
            if c is not None:
                c += c
        for i, inp in enumerate(inps):
            syminp = -inp
            inps[i] = np.concatenate([inp, syminp], axis=0)

    if selection is not None:
        inps = [inps[i] for i in selection]
        color = [color[i] for i in selection]
        markers = [markers[i] for i in selection]
        labels = [labels[i] for i in selection]

    # Initialize key word arguments
    n = len(inps)
    list_of_kwargs = _init_kwargs(
        n, color=color, markers=markers, markersizes=markersizes,
        zorders=zorders, **kwargs
    )

    plt.figure(figsize=figsize)

    max_xtick = xmax
    max_ytick = ymax
    if symmetric:
        plt.xlim((-xmax, xmax))
        plt.xticks(np.linspace(-max_xtick, max_xtick, 5))
    else:
        plt.xlim((0, xmax))
        plt.xticks(np.linspace(0, max_xtick, 3))
    plt.ylim((-ymax, ymax))
    plt.yticks(np.linspace(-max_ytick, max_ytick, 5))

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.grid(which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    for inp, label, kwargs_scatter in zip(inps, labels, list_of_kwargs):
        out = plt.scatter(inp[:, 0], inp[:, 1], label=label, **kwargs_scatter)

    # Chart layout
    kwargs_legend = {}
    if legend_loc is not None:
        kwargs_legend['loc'] = legend_loc
    if labels[0] is not None:
        plt.legend(**kwargs_legend)
    if title is not None:
        plt.title(title, size=fontsize_title)

    return out


def _init_kwargs(
        n, alpha=None, color=None, markers=None, markersizes=None,
        linestyles=None, zorders=None, **kwargs
):
    list_of_kwargs = []
    for i in range(n):
        list_of_kwargs.append(dict(**kwargs))
    if alpha is not None:
        assert len(alpha) == n
        for i, alph in enumerate(alpha):
            list_of_kwargs[i].update(alpha=alph)
    if color is not None:
        assert len(color) == n
        for i, c in enumerate(color):
            list_of_kwargs[i].update(color=c)
    if markers is not None:
        assert len(markers) == n
        for i, m in enumerate(markers):
            list_of_kwargs[i].update(marker=m)
    if markersizes is not None:
        assert len(markersizes) == n
        for i, m in enumerate(markersizes):
            list_of_kwargs[i].update(markersize=m)
    if linestyles is not None:
        assert len(linestyles) == n
        for i, ls in enumerate(linestyles):
            list_of_kwargs[i].update(linestyle=ls)
    if zorders is not None:
        assert len(zorders) == n
        for i, zorder in enumerate(zorders):
            list_of_kwargs[i].update(zorder=zorder)

    return list_of_kwargs


LIST_OF_KWARGS_KERCHARAC = ['spectr_size', 'rgb']

def scatter_characfreqs(
        list_of_inputs, list_of_idx=None, max_normsum=None, min_coherence=None,
        s=80, xlabel=r"$\xi_1$", ylabel=r"$\xi_2$", xmax=np.pi, ymax=np.pi,
        selection=None, discrepancies=None, **kwargs
):
    if not isinstance(list_of_inputs, list):
        list_of_inputs = [list_of_inputs]
        list_of_idx = [list_of_idx]
    n_inps = len(list_of_inputs)
    if list_of_idx is not None:
        assert len(list_of_idx) == n_inps
    else:
        list_of_idx = n_inps * [None]
    kwargs_kercharac = toolbox.extract_dict(LIST_OF_KWARGS_KERCHARAC, kwargs)
    list_of_coords = []
    for inp, idx in zip(list_of_inputs, list_of_idx):
        if not isinstance(inp, pd.DataFrame):
            inp = plots_toolbox.kernel_characterization(
                inp, return_dataframe=True, **kwargs_kercharac
            )
        if idx is not None:
            inp = inp.iloc[idx]
        if max_normsum is not None:
            # remove low-pass filters
            inp = inp[
                inp['norm_sum'] < max_normsum
            ]
        if min_coherence is not None:
            # remove filters with poorly-defined orientation
            inp = inp[
                inp['coherence'] >= min_coherence
            ]
        list_of_coords.append(
            inp[['freq_x', 'freq_y']].to_numpy()
        )

    if len(list_of_coords) == 1:
        list_of_coords = list_of_coords[0]

    out = scatter_centergrid(
        list_of_coords, s=s, xlabel=xlabel, ylabel=ylabel, xmax=xmax, ymax=ymax,
        selection=selection, **kwargs
    )

    # If required, display discrepancies between max pooling and modulus
    if discrepancies is not None:
        cmap = plt.cm.get_cmap('bone')
        cmap = cmap.reversed() # Low values are brighter than high values
        im = plt.imshow(
            discrepancies, cmap=cmap, origin='lower',
            extent=(-np.pi, np.pi, -np.pi, np.pi), vmax=1., alpha=.6
        )
        plt.colorbar(im, shrink=0.9)

    return out


def plot_2d(
        vals, title=None, cmap='bone', figsize=(5, 5), degree_norm=None,
        zero_centered=False, vmin=0., vmax=None,
        xlabel=None, ylabel=None, reversed_cmap=True, colorbar=True,
        shrink_colorbar=1., fontsize_title=FONTSIZE_TITLE, **kwargs
):
    """
    Plot a 2D function.

    Parameters
    ----------
    vals (numpy.ndarray, shape=(height, width))

    """
    if degree_norm is not None:
        kwargs.update(
            norm=PolynomialNorm(degree=degree_norm, vmin=vmin, vmax=vmax)
        )
    elif zero_centered:
        kwargs.update(
            norm=MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0.)
        )
    else:
        kwargs.update(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(cmap)
    if reversed_cmap:
        cmap = cmap.reversed() # Low values are brighter than high values
    plt.figure(figsize=figsize)
    im = plt.imshow(
        vals, cmap=cmap, **kwargs
    )
    if colorbar:
        plt.colorbar(im, shrink=shrink_colorbar)
    if title is not None:
        plt.title(title, size=fontsize_title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    return im


def plot_2dfreqplane(
        vals, fmin=0., fmax=np.pi, xlabel=r"$\xi_1$", ylabel=r"$\xi_2$",
        origin="lower", **kwargs
):
    """
    Plot a 2D function in the frequency plane.

    Parameters
    ----------
    vals (numpy.ndarray, shape=(height, width))

    """
    extent = (fmin, fmax, fmin, fmax)
    im = plot_2d(
        vals, extent=extent, xlabel=xlabel, ylabel=ylabel, origin=origin,
        **kwargs
    )
    return im


def plot_2d_shifts(
        vals, max_shift, xlabel=r"Shift along the $x$-axis",
        ylabel=r"Shift along the $y$-axis", origin="lower", **kwargs
):
    """
    Plot a 2D function in the spatial domain.

    Parameters
    ----------
    vals (numpy.ndarray, shape=(height, width))

    """
    extent = (0., max_shift, -max_shift, 0.)
    im = plot_2d(
        vals, extent=extent, xlabel=xlabel, ylabel=ylabel, origin=origin,
        **kwargs
    )
    return im


def plot_grads(classifs, list_of_out_channels, j, idx=None, **kwargs):
    """
    Plot the mean gradients over all the output channels of a given scale.

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
    **kwargs
        Keyword arguments to be passed as parameters of 'plot_data'.

    """
    grads = plots_toolbox.get_meangrads(classifs, list_of_out_channels, j, idx)
    list_of_x = []
    for grd in grads:
        list_of_x.append(torch.arange(grd.shape[0]))
    out = plot_data(list_of_x, grads, **kwargs)
    plt.xlabel("Batch number")
    plt.ylabel("Mean gradient over the output channels")
    plt.title("Evolution of the gradient at scale {}".format(j))

    return out
