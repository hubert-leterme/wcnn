"""
Contains custom metrics and a dictionary named METRICS_DICT.
METRICS_DICT is organized as follows:
- keys (str): metrics names, which are possible values for the 'metrics'
    constructor parameter of classifier.WaveCnnClassifier.
- values (tuple): tuples containing:
    (a) a callable representing the metrics itsself;
    (b) a dictionary of keyword arguments for the specified metrics.

"""
import numpy as np
import torch
from sklearn import metrics

def top_N_accuracy_score(
        y_true, y_pred, N=1, backend='torch', get_item=True
):
    """
    Compute the accuracy score for the top-N results.

    Parameters
    ----------
    y_true (torch.Tensor)
        Truth vector, represented as a vector of length K, where K is the number of
        evaluation examples.
    y_pred (torch.Tensor)
        Tensor of shape (K, num_classes). The (i, j) element represents the probability of
        example i to belong to the j-th class.
    N (int, default=1)
        If y_true[i] is one of the top-N classes for example i, then it is considered as
        correclty classified.
    backend (str, default='torch')
    get_item (bool, default=True)
        Whether to return a float instead of a one-element tensor. Set to False in case of
        distributed evaluation (we need a PyTorch tensor to gather results from all GPUs).

    """
    if backend == 'torch':
        with torch.no_grad():
            _, top_classes = y_pred.topk(N, 1, True, True)
            top_classes = top_classes.T # Shape = (N, K)
            differing_labels = torch.prod(top_classes - y_true, dim=0)
            score = (differing_labels == 0)
            ret = torch.sum(score) / score.numel()
            if get_item:
                ret = ret.item()

    else:
        top_classes = np.argsort(y_pred, 1)[:, -N:].T # Shape = (N, K)
        differing_labels = np.prod(top_classes - y_true, axis=0)
        score = (differing_labels == 0)
        ret = np.sum(score) / score.size
        if not get_item:
            raise ValueError("Backend must be set to 'torch' if 'get_item' is False.")

    return ret


def list_of_accuracy_scores(y_true, y_pred, tops=None, **kwargs):
    """
    Compute accuracy score for the top-N results, for several values of N.

    Parameters
    ----------
    y_true (torch.Tensor)
        Truth vector, represented as a vector of length K, where K is the number of
        evaluation examples.
    y_pred (torch.Tensor)
        Tensor of shape (K, num_classes). The (i, j) element represents the probability of
        example i to belong to the j-th class.
    tops (list, default=None)
        List of tops. See top_N_accuracy_score for more details.
    backend (str, default='torch')
    get_item (bool, default=True)
        Whether to return a float instead of a one-element tensor. Set to False in case of
        distributed evaluation (we need a PyTorch tensor to gather results from all GPUs).

    """
    res = dict()
    for N in tops:
        res[N] = top_N_accuracy_score(y_true, y_pred, N=N, **kwargs)

    return res


def mean_flip_rate(y_pred, max_step_size, get_item=True):
    """
    Compute mean flip rate between several outputs, corresponding to an input image
    and its variations (shifts, perturbations...). From D. Hendrycks and T. Dietterich,
    “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations,” in
    ICLR, 2019. doi: 10.48550/arXiv.1903.12261.
    Authors' original code can be found here: https://github.com/hendrycks/robustness

    Parameters
    ----------
    y_pred (torch.Tensor)
        Tensor of shape (K, n_samples), where K denotes the number of images in the dataset
        and n_samples denotes the number of variants for each image. The tensor contains
        top-1 predicted labels for each output.
    max_step_size (int)
        Maximum distance to be considered between frames.
    get_item (bool, default=True)
        Whether to return a float instead of a one-element tensor. Set to False in case of
        distributed evaluation (we need a PyTorch tensor to gather results from all GPUs).

    """
    out = {}
    for step_size in range(1, max_step_size + 1):
        # Compute mean flip rate for any 2 frames with distance given by step_size
        # Shape = (K, (n_samples - step_size))
        comps = y_pred[:, :-step_size] != y_pred[:, step_size:]
        comps = comps.type(torch.get_default_dtype())
        mfr = torch.mean(comps) # Average over images and samples
        if get_item:
            mfr = mfr.item()
        out[step_size] = mfr # Update result dictionary

    return out



METRICS_DICT = {
    'accuracy_score': (metrics.accuracy_score, {}),
    'average_precision_score':(
        metrics.average_precision_score, dict(average='micro')
    ),
    'top1_accuracy_score': (top_N_accuracy_score, dict(N=1)),
    'top5_accuracy_score': (top_N_accuracy_score, dict(N=5)),
    'top1_5_accuracy_scores': (
        list_of_accuracy_scores, dict(tops=[1, 5])
    )
}
