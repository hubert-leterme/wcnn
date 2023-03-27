import PIL
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

#=================================================================================
# Operations on tensors
#=================================================================================

def convolve_kernels(
        in_channels, list_of_kernels, list_of_strides,
        list_of_dilations=None, verbose=False
):
    """
    Forward-propagating an image through a series of convolution layers (without
    activation function in between) is equivalent to propagate the same image
    through a single convolution layer, whose characteristics are given by
    the outputs of this function.

    Parameters
    ----------
    in_channels (int)
        Number of input channels.
    list_of_kernels (list of torch.tensor)
        List of 2D convolution kernels of shape
            (out_channels, in_channels/groups, n, n).
        For each kernel in the list, a (possibly decimated) convolution is
        applied to the input and the result is passed to the next kernel. For
        instance a Db3 convolution kernel has shape (4, 1, 6, 6), i.e. four
        convolution filters of size (6, 6) applied with decimation to each input
        channel (e.g. RGB channels) separately.
        N.B.: The number of groups is automatically inferred from the shape of the
        kernels (see torch.nn.Conv2d for more details).
    list_of_strides (list of int/tuple)
        Stride (decimation factor) for each convolution kernel.
    list_of_dilations (list of int/tuple, default=None)
        Dilation for each convolution kernel. If none is given, the it is set to 1.
    verbose (bool, default=False)

    Returns
    -------
    out_kernel, out_stride (torch.tensor, int)
        The resulting convolution kernel and stride (decimation factor). Without
        loss of generality, we consider that the overall convolution is performed
        with only one group (each output channel is influenced by all the input
        channels).

    """
    n_layers = len(list_of_kernels)
    assert len(list_of_strides) == n_layers
    if list_of_dilations is not None:
        assert len(list_of_dilations) == n_layers
    else:
        list_of_dilations = n_layers * [1]

    if verbose:
        print("===================================================")
        print("Composition of the following convolution operators:\n")

    # Initialization
    out = get_identity_convkernel(in_channels)
    resulting_stride = (1, 1)

    # Iterate over the list of kernels
    for ker, stride, dilation in zip(
            list_of_kernels, list_of_strides, list_of_dilations
    ):
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if verbose:
            print("Kernel of shape {}".format(ker.shape))
            print("Stride = {}".format(stride))
            print("Dilation = {}".format(dilation))
            print("")

        # Number of groups
        assert in_channels % ker.shape[1] == 0
        groups = in_channels // ker.shape[1]
        assert ker.shape[0] % groups == 0
        in_channels = ker.shape[0] # For the next kernel

        # Convolution with the current kernel
        dilation0 = (
            resulting_stride[0] * dilation[0],
            resulting_stride[1] * dilation[1]
        )
        out = _convolve(out, ker, dilation=dilation0, groups=groups)

        # Inflate the dilation factor for the next convolution
        resulting_stride = (
            resulting_stride[0] * stride[0],
            resulting_stride[1] * stride[1]
        )

    # Convert the output (minibatch of feature maps) into a convolution kernel.
    # See 'Documents/Bibliography/Proofs/Cascading convolutions'
    out = torch.flip(out, dims=(-2, -1))
    out = out.permute(1, 0, 2, 3) # Permute input and output channels

    return out, resulting_stride


def _convolve(x, y, dilation=(1, 1), groups=1, do_not_switch=False):

    def _get_padding(ker):
        ker_size_with_dilation = tuple(
            (dil * (m-1) + 1) for dil, m in zip(dilation, ker.shape[-2:])
        )
        # Padding must be even
        pad = tuple(-(-m0//2) * 2 for m0 in ker_size_with_dilation)
        return pad

    size_x = x.shape[-2] * x.shape[-1]
    size_y = y.shape[-2] * y.shape[-1]

    if do_not_switch or (dilation != (1, 1)) or (groups != 1) \
            or (size_y <= size_x):
        pad = _get_padding(y)
        out = F.conv2d(
            x, y, padding=pad, dilation=dilation, groups=groups
        )
    else: # Switch input and convolution kernel, for computational purpose
        pad = _get_padding(x)
        out = F.conv2d(y, x, padding=pad)
        out = torch.flip(out, dims=(-2, -1))
        out = out.permute(1, 0, 2, 3) # Permute input and output channels

    return out


def get_equivalent_convkernel(func, in_channels, *args, **kwargs):
    """
    Several forward methods implemented in this package involve selection,
    duplication or recombination of feature maps. These operations can be
    mathematically written in the form of a 1x1 multichannel convolution
    operator.
    Given such an operation, the present function computes its equivalent
    convolution kernel. Note that it only works on unbiased linear operations.

    Parameters
    ----------
    f (callable)
        Function mapping PyTorch tensors X to Y, where the shapes of X and Y are
            (n_imgs, in_channels, height, width)
        and
            (n_imgs, out_channels, height, width),
        respectively. f must be an unbiased, pointwise and linear operation.
    in_channels (int)
        Number of input channels of f. The number of output channels is automa-
        tically deduced from in_channels and f.
    *args, **kwargs
        Positional and keyword arguments of f.

    Returns
    -------
    The resulting kernel, which is a PyTorch tensor of shape (out_channels,
    in_channels, 1, 1).

    """
    # Initialization (identity convolution kernel)
    out = get_identity_convkernel(in_channels)

    # Forward-propagate the input
    out = func(out, *args, **kwargs)

    # Convert the output (minibatch of feature maps) into a convolution kernel.
    # See 'Documents/Bibliography/Proofs/Cascading convolutions'
    out = torch.flip(out, dims=(2, 3))
    out = out.permute(1, 0, 2, 3) # Permute input and output channels

    return out


def get_padding(
        strides, kersizes, dilations=None, even_padding=True
):
    """
    Given a series of strided convolutions, compute the padding to apply to the
    input in order to get approximately:
    size(output) = size(intput) / prod(strides)

    """
    if not isinstance(strides, list):
        strides = [strides]
        if dilations is not None:
            dilations = [dilations]
        kersizes = [kersizes]

    if dilations is None:
        dilations = len(strides) * [1]

    def _incr_padding(stride, kersize, dilation, prod_strides):
        dilated_kersize = dilation * (kersize - 1) + 1
        return prod_strides * (dilated_kersize - stride + 1)

    out_vert, out_hori = (0, 0)
    prod_strides_vert, prod_strides_hori = (1, 1)
    for s, k, d in zip(strides, kersizes, dilations):
        svert, shori = get_tuple(s)
        kvert, khori = get_tuple(k)
        dvert, dhori = get_tuple(d)

        out_vert += _incr_padding(
            svert, kvert, dvert, prod_strides_vert
        )
        prod_strides_vert *= svert

        out_hori += _incr_padding(
            shori, khori, dhori, prod_strides_hori
        )
        prod_strides_hori *= shori

    out_vert //= 2
    out_hori //= 2

    # Make padding even if necessary
    if even_padding:
        out_vert = round(out_vert // 2) * 2
        out_hori = round(out_hori // 2) * 2

    return out_vert, out_hori


def unsqueeze_vertical_kernels(vweight):

    assert len(vweight.shape) == 2
    vweight = torch.unsqueeze(vweight, -1) # Tensor of shape (in_channels, 3, 1)
    vweight = torch.unsqueeze(vweight, 1) # Tensor of shape (in_channels, 1, 3, 1)

    return vweight


def unsqueeze_horizontal_kernels(hweight):

    assert len(hweight.shape) == 2
    hweight = torch.unsqueeze(hweight, -2) # Tensor of shape (in_channels, 1, 3)
    hweight = torch.unsqueeze(hweight, 1) # Tensor of shape (in_channels, 1, 1, 3)

    return hweight


def get_tuple(inp):
    if not isinstance(inp, tuple) or isinstance(inp, list):
        inp = (inp, inp)
    return inp


def separable_conv2d(x, vweight, hweight, stride, padding=0, dilation=1):

    # Vertical part of the separable 2D convolution
    x = F.conv2d(
        x, vweight, stride=(stride, 1), padding=(padding, 0),
        dilation=(dilation, 1), groups=x.shape[1]
    )
    # Horizontal part of the separable 2D convolution
    x = F.conv2d(
        x, hweight, stride=(1, stride), padding=(0, padding),
        dilation=(1, dilation), groups=x.shape[1]
    )
    return x


def get_real_part(x):
    assert x.shape[1] % 2 == 0
    return x[:, ::2]


def get_modulus(x):

    n_imgs, in_channels, height, width = x.shape
    assert in_channels % 2 == 0

    # Shape (n_imgs, in_channels / 2, 2, height, width)
    x = x.view(n_imgs, -1, 2, height, width)

    # Shape (n_imgs, in_channels / 2, width, height, 2)
    # Apply .contiguous() to avoid RuntimeError (see https://bit.ly/2WFn6DD)
    x = x.permute(0, 1, 3, 4, 2).contiguous()

    # Complex tensor, shape (n_imgs, in_channels / 2, width, height)
    x = torch.view_as_complex(x)

    return torch.abs(x)


def find_smallest_size(list_of_tensors):
    out = [float('inf'), float('inf')]
    for x in list_of_tensors:
        size = x.shape[-2:]
        for i, (dim, dim0) in enumerate(zip(size, out)):
            if dim < dim0:
                out[i] = dim
    return out


def find_largest_size(list_of_tensors):
    out = [0, 0]
    for x in list_of_tensors:
        size = x.shape[-2:]
        for i, (dim, dim0) in enumerate(zip(size, out)):
            if dim > dim0:
                out[i] = dim
    return out


def crop_featmaps_before_concat(list_of_featmaps):
    size0 = find_smallest_size(list_of_featmaps)
    for i, y0 in enumerate(list_of_featmaps):
        list_of_featmaps[i] = TF.center_crop(y0, size0)


def pad_featmaps_before_concat(list_of_featmaps):

    size0 = find_largest_size(list_of_featmaps)
    for i, y0 in enumerate(list_of_featmaps):
        size = y0.shape[-2:]
        padding_needed = False
        pad = []
        for s0, s in zip(size0, size):
            pad_left = (s0 - s) // 2
            pad_right = -((s - s0) // 2)
            pad += [pad_left, pad_right]
            if s < s0:
                padding_needed = True
        if padding_needed:
            list_of_featmaps[i] = F.pad(y0, pad)


def get_identity_convkernel(in_channels):
    return torch.eye(in_channels).view(in_channels, in_channels, 1, 1)


def crop_edges(x, n_pixels):
    return x[..., n_pixels:-n_pixels, n_pixels:-n_pixels]


def extract_patches(
        inp, size, n_patches, beg_v=0, beg_h=0, end_v=None, end_h=None,
        direction=None, return_ref=False
):
    out = []

    if isinstance(size, int):
        size = (size, size)
    sizev, sizeh = size

    if end_v is None or end_v > n_patches:
        end_v = n_patches
    if end_h is None or end_h > n_patches:
        end_h = n_patches

    if isinstance(inp, PIL.Image.Image):
        if direction is None:
            for i in range(beg_v, end_v):
                for j in range(beg_h, end_h):
                    out.append(inp.crop((j, i, sizeh + j, sizev + i)))
        elif direction == "vertical":
            for i in range(beg_v, end_v):
                out.append(inp.crop((beg_h, i, sizeh + beg_h, sizev + i)))
        elif direction == "horizontal":
            for j in range(beg_h, end_h):
                out.append(inp.crop((j, beg_v, sizeh + j, sizev + beg_v)))
        elif direction == "diagonal":
            for k in range(n_patches):
                out.append(inp.crop((
                    beg_h + k, beg_v + k, sizeh + beg_h + k, sizev + beg_v + k
                )))

    elif isinstance(inp, torch.Tensor):
        if direction is None:
            for i in range(beg_v, end_v):
                for j in range(beg_h, end_h):
                    out.append(inp[..., i:(sizev + i), j:(sizeh + j)])
        elif direction == "vertical":
            for i in range(beg_v, end_v):
                out.append(inp[..., i:(sizev + i), beg_h:(sizeh + beg_h)])
        elif direction == "horizontal":
            for j in range(beg_h, end_h):
                out.append(inp[..., beg_v:(sizev + beg_v), j:(sizeh + j)])
        elif direction == "diagonal":
            for k in range(n_patches):
                out.append(inp[
                    ...,
                    (beg_v + k):(sizev + beg_v + k),
                    (beg_h + k):(sizeh + beg_h + k)
                ])
        out = torch.stack(out)

    else:
        raise TypeError

    if return_ref:
        out = (out, inp)

    return out


#=================================================================================
# Operations on PyTorch modules
#=================================================================================

def freeze_layers(convs, fcs=None, frozen_modules=0, frozen_fc=0):
    """
    Freeze layers.

    """
    if frozen_fc > 0:
        frozen_modules = len(convs) # All convolution layers are frozen

    # Freeze fully-connected layers
    for i in range(frozen_fc):
        fc = fcs[i]
        for param in fc.parameters():
            param.requires_grad = False

    # Freeze convolution layers
    for i in range(frozen_modules):
        conv = convs[i]
        for param in conv.parameters():
            param.requires_grad = False


def get_extra_repr(*args, **kwargs):

    msg = ""
    for val in args:
        msg += "{}, ".format(val)
    for key in kwargs:
        msg += "{}={}, ".format(key, kwargs[key])

    return msg[:-2]


def get_n_trainable_params(net):
    """
    Returns the number of trainable parameters, i.e., those with attribute
    `requires_grad` set to `True`.

    """
    out = 0
    for param in net.parameters():
        if param.requires_grad:
            out += param.numel()
    return out


def extract_subnet(net, submodule=None, del_modules=None, depth=None):

    if submodule is not None:
        for elt in submodule:
            net = getattr(net, elt)
    if del_modules is not None:
        for elt in del_modules:
            net.__delattr__(elt)
    if depth is not None:
        net = net[:depth]

    return net


def merge_sections(split_sections, types):

    split_sections0 = []
    types0 = []
    for outsec, wch in zip(split_sections, types):
        try:
            isnewtype = (wch != types0[-1])
        except IndexError:
            isnewtype = True
        if isnewtype:
            split_sections0.append(0)
            types0.append(wch)
        split_sections0[-1] += outsec

    return split_sections0, types0


def get_model_size(model):
    """
    See discussion at https://discuss.pytorch.org/t/finding-model-size/130275

    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return param_size + buffer_size


def compute_img_gradient(inp, axis):
    """
    Compute the gradient of a minibatch of images by using a Sobel filter.

    Parameters
    ----------
    inp (torch.Tensor)
        Input tensor of shape (n_imgs, in_channels, n, m)
    axis (int)
        Wether to compute the gradient along the x axis (axis='x') or along the
        y axis (axis='y')

    Returns
    -------
    out (torch.Tensor)
        Output gradient tensor of shape (n_imgs, in_channels, n-2, m-2)

    """
    gradient = torch.Tensor([-1, 0, 1])
    smoothing = torch.Tensor([1, 2, 1])
    in_channels = inp.shape[1]
    if axis == 'x':
        vweight = in_channels * [smoothing] # Vertical smoothing
        hweight = in_channels * [gradient] # Horizontal gradient
    elif axis == 'y':
        vweight = in_channels * [gradient] # Vertical gradient
        hweight = in_channels * [smoothing] # Horizontal smoothing
    else:
        raise ValueError("'axis' must be equal to 'x' or 'y'.")

    vweight = torch.stack(vweight) # Tensor of shape (in_channels, 3)
    vweight = unsqueeze_vertical_kernels(vweight)

    hweight = torch.stack(hweight) # Tensor of shape (in_channels, 3)
    hweight = unsqueeze_horizontal_kernels(hweight)

    out = separable_conv2d(
        inp, vweight, hweight, stride=1, padding=0
    )

    return out


def structure_tensor(inp):
    """
    Compute the structure tensor as described in [Jahne, B. (2004). Practical
    Handbook on Image Processing for Scientific and Technical Applications. CRC
    Press]. The output is not exactly the structure tensor itself, but a
    complete description of it using a 3D vector for each image and each channel:
    [
        J00 + J11
        J11 - J00
        2 J01
    ]
    where J denotes the 2x2 symmetric structure tensor.

    Parameter
    ---------
    inp (torch.Tensor)
        Input tensor of shape (n_imgs, in_channels, n, m)

    Returns
    -------
    out (torch.Tensor)
        Tensor structure of shape (n_imgs, in_channels, 3)

    """
    # Compute partial derivatives
    grad_x = compute_img_gradient(inp, axis='x')
    grad_y = compute_img_gradient(inp, axis='y')

    # Compute structure tensor
    Jxx = torch.sum(grad_x * grad_x, dim=(-2, -1))
    Jxy = torch.sum(grad_x * grad_y, dim=(-2, -1))
    Jyy = torch.sum(grad_y * grad_y, dim=(-2, -1))

    out = torch.stack([
        Jxx + Jyy,
        Jyy - Jxx,
        2 * Jxy
    ], dim=-1)

    return out


def _get_angles(struct, convert_to_degrees=False):

    tan_twotheta = struct[:, :, 2] / struct[:, :, 1]
    pishift = (
        torch.sign(struct[:, :, 1] + struct[:, :, 2] * tan_twotheta) + 1
    ) / 2 # 0 or 1
    out = (torch.atan(tan_twotheta) + pishift * torch.pi) / 2
    if convert_to_degrees:
        out *= 180 / torch.pi
    return out


def homogeneity_direction_coherence(struct):
    """
    Parameters
    ----------
    struct (torch.Tensor)
        Tensor of shape (n_imgs, in_channels, 3), describing a bunch of structure
        tensors. See structure_tensor for more information.

    Returns
    -------
    homogeneity (torch.Tensor)
        Tensor of shape (n_imgs, in_channels) indicating the homogeneity of the
        considered images. It is equal to J00 + J11, i.e., the sum of eigenvalues
        of the structure tensor. The smaller, the more homogeneous.

    direction (torch.Tensor)
        Tensor of shape (n_imgs, in_channels, 2) containing normalized vectors
        indicating the main orientation of each feature map.

    coherence (torch.Tensor)
        Tensor of shape (n_imgs, in_channels) containing the coherence indice.

    See [B. Jahne, Practical Handbook on Image Processing for Scientific and
    Technical Applications. CRC Press, 2004.] (p. 419) for more details.

    """
    homogeneity = struct[:, :, 0]

    angles = _get_angles(struct)
    nx = torch.cos(angles)
    ny = torch.sin(angles)
    direction = torch.stack([nx, ny], dim=-1)

    eigendiff = torch.sqrt(struct[:, :, 1]**2 + struct[:, :, 2]**2)
    coherence = eigendiff / homogeneity

    return homogeneity, direction, coherence


def get_freqs(s):
    """
    Generate a grid of frequency coordinates.

    """
    s1, s2 = s
    freqs_1d_x = torch.fft.fftfreq(s2)
    freqs_1d_y = -torch.fft.fftfreq(s1)
    freqs_x, freqs_y = torch.meshgrid(freqs_1d_x, freqs_1d_y, indexing='xy')
    freqs = torch.stack([freqs_x, freqs_y], dim=-1)

    return freqs


def get_analytic_kernel(weight, s=None, fft=False):
    """
    Compute the 2D Hilbert transform of real-valued filters, and return complex filters
    whose energy is located in one half of the Fourier domain.

    Parameters
    ----------
    weight (torch.Tensor)
        PyTorch tensor of shape (out_channels, in_channels, height, width), containing
        out_channels * in_channels convolution kernels.
    s (tuple, default=None)
        Signal size in the transformed dimensions, when calling torch.fft.fft2
    fft (bool, default=True)
        Whether to return the complex-valued filters in the Fourier domain. If set to False,
        then spatial filters (cropped to the input shape) are returned.

    Returns
    -------
    out (torch.Tensor)
        Complex tensor of shape (out_channels, in_channels, height, width) is spatial=True,
        (out_channels, in_channels, height_fft, width_fft) otherwise.
    
    """
    _, in_channels, height, width = weight.shape
    if s is None:
        s = (height, width)

    # Compute the mean value over the input channels, to get the main orientation
    avgweight = torch.mean(weight, dim=1).unsqueeze(1)

    # Compute eigenvalues of the tensor structure
    struct = structure_tensor(avgweight)

    # Estimate the principal orientation of each filter
    _, maindir, _ = homogeneity_direction_coherence(struct)

    # Generate a grid of frequency coordinates
    freqs = get_freqs(s)

    # Compute inner products between frequency vectors and main orientation
    innerprods = torch.inner(maindir, freqs)

    # Mask to compute the complex-valued kernels: half of the Fourier domain is set to 0
    mask_fft = 2. * (innerprods >= 0)
    mask_fft = mask_fft.repeat(1, in_channels, 1, 1)

    # Compute the FFT of the real-valued filters and deduce the FFT of the complex filters
    weight_fft = torch.fft.fft2(weight, s=s)
    complweight_fft = mask_fft * weight_fft

    if not fft:
        out = torch.fft.ifft2(complweight_fft)[:, :, :height, :width]
    else:
        out = complweight_fft

    return out
