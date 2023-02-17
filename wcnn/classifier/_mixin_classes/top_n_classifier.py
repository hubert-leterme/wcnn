import warnings

import torch
import torch.distributed as dist

from .. import classifier_toolbox
from ... import toolbox
from ... import metrics as metr

from ..classifier_toolbox import LIST_OF_KWARGS_FORWARDEVAL

LIST_OF_KWARGS_SAVE = ['path_to_classif']

class TopNClassifierMixin:

    def score(
            self, dataset, eval_mode='onecrop', metrics=None, max_step_size=5,
            distributed=False, gpu=None, rank=None, world_size=None, save_results=False,
            ds_name=None, split=None, commit=None, overwrite=False, **kwargs
    ):
        """
        Return the score on the given test data and labels, according to the
        'metrics' constructor parameter.

        Parameters
        ----------
        dataset (instance of any class defined in wcnn.data.datasets)
        eval_mode (str, default='onecrop')
            'onecrop': the smaller side of each image is resized to the size
                specified as constructor parameter; then a center crop is
                applied to get a square image.
            'tencrops': outputs are averaged over 10 square patches, the size of
                which is specified as constructor parameter (4 corner crops and
                one center crop applied on each image and its horizontal flipped
                version).
            'shifts': compute predictions for several shifted versions of the
                same image.
            'perturbations': compute predictions for several perturbed versions
                same image. Only works if 'dataset' is an instance of
                data.datasets.ImageNetP.
        max_shift (int, default=None)
            Maximum number of pixels in each direction, for image shift. Must be
            provided if 'mode' is set to 'shifts'.
        pixel_divide (int, default=1)
            Number of intermediate shifts per pixel. For example, if set to 2, then
            images will be shifted by steps of 0.5 pixels. This is made possible by
            using bilinear interpolation and antialiased downsampling. See
            https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html
            for more details.
        shift_direction (str, default=None)
            One of 'vertical', 'horizontal' or 'diagonal'. If none is given, then shifts
            will be computed along both directions, which may lead to high memory
            consumption.
        metrics (str, default=None)
            One of 'kldiv' (KL divergence between outputs) or 'mfr' (mean flip rate between
            any two frames with distance ranging from 1 to max_step_size---see below).
            If none is given, then the metrics is chosen as follows:
            - eval_mode == 'onecrop' or 'tencrops' >> provided by attribute self.metrics;
            - eval_mode == 'shifts' or 'perturbations' >> mean flip rate (equivalent to
              metrics='mfr').
        max_step_size (int, default=5)
            Step size when metrics == 'mfr' (see above).
        distributed (bool, default=False)
            Whether to evaluate on several GPUs.
        gpu (int, default=None)
            GPU on which data are loaded.
        rank (int, default=None)
            To be provided if the number of nodes is greater than one. If none is
            given, then it is automatically set to the value given by 'gpu'.
        world_size (int, default=None)
            Total number of GPUs (= n_nodes * N_GPUS). If none is given, then it is
            automatically set to N_GPUS.
        save_results (bool, default=False)
            Whether to save the returned values.
        ds_name (str, default=None)
            Type of the dataset. Only used if 'save_results' is set to True.
        split (str, default=None)
            Split of the dataset. Only used if 'save_results' is set to True.
        commit (str, default=None)
            Hash of the current git commit. Only used if 'save_results' is set to True.
        overwrite (bool, default=False)
            Whether to overwrite existing recorded score. Only used if 'save_results'
            is set to True.
        path_to_classif (str, default=None)
            Directory in which to save the updated classifier. This argument should be
            provided in debug mode, to avoid saving fake data. Only used if 'save_results'
            is set to True.

        Returns
        -------
        out (float or dict)
            The score, whose type depends on the specified metrics.

        """
        if distributed:
            gpu, rank, world_size = classifier_toolbox.get_gpu_rank(
                gpu, rank, world_size
            )
            kwargs_reduce = {}
        self.infoprinter_ = toolbox.InfoPrinter(
            distributed=distributed, gpu=gpu, verbose=self.verbose
        )

        if metrics is None:
            metrics = classifier_toolbox.get_default_metrics(self, eval_mode)

        # Retrieve existing dictionary of scores, if requested
        if save_results:
            if (not distributed) or (rank == 0):
                eval_mode0 = eval_mode
                if eval_mode0 == 'shifts':
                    shift_direction = kwargs.get('shift_direction')
                    if shift_direction is not None:
                        eval_mode0 = '_'.join([eval_mode0, shift_direction])
                scoredict = self._get_scoredict(
                    ds_name, split, eval_mode0, metrics, overwrite
                )
                kwargs_score = {}
                if commit is not None:
                    kwargs_score.update(commit=commit)
                kwargs_save = toolbox.extract_dict(LIST_OF_KWARGS_SAVE, kwargs, pop=False)

        kwargs_predict = toolbox.extract_dict(LIST_OF_KWARGS_FORWARDEVAL, kwargs, pop=False)
        self.infoprinter_("Forwarding data...")

        # Set compute_score = True; otherwise the results from GPUs will be
        # gathered in a single tensor.
        output, ytrue = self.predict(
            dataset, eval_mode=eval_mode, metrics=metrics, compute_score=True,
            distributed=distributed, gpu=gpu, rank=rank, world_size=world_size,
            **kwargs_predict
        )
        self.infoprinter_("...done.")

        if metrics == 'mfr':
            kwargs_metrics = dict(max_step_size=max_step_size)
            if distributed:
                # The output of the score evaluation must be a PyTorch tensor
                # for all_reduce method
                kwargs_metrics.update(get_item=False)
                kwargs_reduce.update(get_item=True)
            out = metr.mean_flip_rate(output, **kwargs_metrics)

        elif metrics == 'kldiv':
            # The KL divergence has already been computed
            # Compute mean over evaluation examples
            out = output.mean(dim=0)

        else:
            # The metrics is determined by attribute self.metrics_
            kwargs_metrics = self.args_metrics_.copy()
            if distributed:
                if not self.metrics_ in (
                    metr.top_N_accuracy_score, metr.list_of_accuracy_scores
                ):
                    # TODO The score must be computed on the GPU
                    raise NotImplementedError
                # The output of the score evaluation must be a PyTorch tensor
                # for all_reduce method
                kwargs_metrics.update(get_item=False)
                kwargs_reduce.update(get_item=True)

            # Score computed on the matrix of probabilities
            self.infoprinter_("Computing scores...")
            out = self.metrics_(
                ytrue, output, **kwargs_metrics
            )
            self.infoprinter_("...done.")

        if distributed:
            self.infoprinter_("Gathering scores from all GPUs...")
            if not isinstance(out, dict):
                out = _get_avgscore_form_gpus(out, len(ytrue), **kwargs_reduce)
            else: # 'out' is a dictionary of scores
                for key, score in out.items():
                    out[key] = _get_avgscore_form_gpus(score, len(ytrue), **kwargs_reduce)
            self.infoprinter_("...done.")

        # Save results (if required)
        if save_results:
            if (not distributed) or (rank == 0):
                scoredict[metrics] = classifier_toolbox.Score(out, **kwargs_score)
                self.save(status='snapshot', save_state_dicts=False, **kwargs_save)
            self.infoprinter_("Dataset:", ds_name, always=True)
            self.infoprinter_("Split:", split, always=True)
            self.infoprinter_("Evaluation mode:", metrics, always=True)
            self.infoprinter_(scoredict[metrics], always=True)

        return out


    def _get_scoredict(
            self, ds_name, split, eval_mode0, metrics, overwrite
    ):
        scoredict = self.eval_scores_
        for key in ds_name, split, eval_mode0:
            try:
                obj = scoredict[key]
                assert isinstance(obj, dict)
            except KeyError:
                scoredict[key] = {}
                scoredict = scoredict[key]
            except AssertionError as exc: # Backward compatibility
                if key == eval_mode0 and 'shifts' in eval_mode0:
                    scoredict[key] = {}
                    scoredict = scoredict[key]
                    scoredict['kldiv'] = obj
                elif key == eval_mode0 and eval_mode0 in ('onecrop', 'tencrops'):
                    scoredict[key] = {}
                    scoredict = scoredict[key]
                    scoredict['top1_5_accuracy_scores'] = obj
                else:
                    raise ValueError(
                        "{}[{}] is expected to be a dictionary.".format(scoredict, key)
                    ) from exc
            else:
                scoredict = obj
                if key == eval_mode0:
                    # Backward compatibility
                    try:
                        scoredict['top1_5_accuracy_scores'] = \
                            scoredict.pop('list_of_accuracy_scores')
                    except KeyError:
                        pass

        registered_score = scoredict.get(metrics)
        if registered_score is not None:
            msg = "A score already exists for this dataset, split, evaluation mode and metrics.\n"
            msg += str(registered_score)
            if not overwrite:
                msg += "\nSet 'overwrite' to True to replace the old score by a new one."
                self.infoprinter_(msg, always=True)
                raise ValueError(msg)
            else:
                msg += "\nThis score will be overwritten."
                self.infoprinter_(msg, always=True)
                warnings.warn(msg)

        return scoredict


def _get_avgscore_form_gpus(val, count, get_item=False):

    assert isinstance(val, torch.Tensor)
    sumval = count * val
    count = torch.tensor(count, device=val.device)
    dist.all_reduce(sumval, dist.ReduceOp.SUM, async_op=False)
    dist.all_reduce(count, dist.ReduceOp.SUM, async_op=False)

    out = sumval / count.item()
    if get_item:
        out = out.item()

    return out
