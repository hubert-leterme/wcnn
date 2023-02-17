import time

import torch
from torch.utils import data
from torch.nn import functional as F
from torchvision.datasets import vision

from ... import transforms, toolbox
from ...cnn import cnn_toolbox

from .. import classifier_toolbox

from ..classifier_toolbox import LIST_OF_KWARGS_FORWARDEVAL

# Fix bug 'RuntimeError: received 0 items of ancdata'
# See https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/3
torch.multiprocessing.set_sharing_strategy('file_system')

class PredictorMixin:

    def predict(
            self, dataset, eval_mode='onecrop', metrics=None,
            distributed=False, gpu=None, rank=None, world_size=None, **kwargs
    ):
        """
        The output of the network is transformed through softmax (the sum of
        each values in the output tensor is equal to 1).

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
                same image. Do not work if 'dataset' is an instance of
                data.datasets.ImageNetP.
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
            any two frames with a given distance; see method WaveCnnClassifier.score for more
            details).
            If none is given, then the metrics is chosen as follows:
            - eval_mode == 'onecrop' or 'tencrops' >> provided by attribute self.metrics;
            - eval_mode == 'shifts' or 'perturbations' >> mean flip rate (equivalent to
              metrics='mfr').
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

        Returns
        -------
        ouput (torch.Tensor)

            |------------------|---------------------|-----------------|
            | Metrics          | Output content      | Tensor shape    |
            |------------------|---------------------|-----------------|
            | None             | Probability vectors | (K, Q)          |
            | 'mfr'            | Predicted labels    | (K, S)          |
            | 'kldiv'          | KL divergence       | (K, S)          |
            |------------------|---------------------|-----------------|

            where K denotes the number of images in the evaluation set, Q denotes the
            number of classes and S denotes the number of samples for each image (number
            of shifts or number of perturbations).

        ytrue (torch.Tensor)
            Ground truth labels, with shape (K,).

        """
        if distributed:
            gpu, rank, world_size = classifier_toolbox.get_gpu_rank(
                gpu, rank, world_size
            )
        self.infoprinter_ = toolbox.InfoPrinter(
            distributed=distributed, gpu=gpu, verbose=self.verbose
        )

        # Transform to be applied to the output of the forward-propagation
        if metrics is None:
            metrics = classifier_toolbox.get_default_metrics(self, eval_mode)

        init_transf = lambda x: _softmax(x, eval_mode)
        if metrics is None or "accuracy_score" in metrics:
            transf = init_transf
        elif metrics == 'mfr':
            transf = lambda x: _compute_ypred(x, init_transf=init_transf)
        elif metrics == 'kldiv':
            transf = lambda x: _compute_kldiv(x, init_transf=init_transf)
        else:
            raise ValueError("Wrong value for argument 'metrics'.")

        # Forward propagate the data
        kwargs_forward = toolbox.extract_dict(LIST_OF_KWARGS_FORWARDEVAL, kwargs)
        output, ytrue = self._forward_with_no_grad(
            dataset=dataset, eval_mode=eval_mode, output_transform=transf,
            distributed=distributed, gpu=gpu, world_size=world_size,
            **kwargs_forward
        )

        return output, ytrue


    def _forward_with_no_grad(
            self, dataset, eval_mode, distributed, gpu, world_size,
            output_transform=None, return_ref=False, compute_tail=True,
            batch_size=None, verbose_print_batch=None,
            workers=None, size=None, **kwargs
    ):
        # Initialize transform for the output
        if output_transform is None:
            output_transform = lambda x: (x, None) # Identity (no prediction)

        # Initialize parameters
        if batch_size is None:
            batch_size = self.batch_size_test
        if verbose_print_batch is None:
            verbose_print_batch = self.verbose_print_batch
        if size is None:
            size = self.size

        # Transform data with compression and conversion to PyTorch tensors
        if not dataset.ready_for_use:
            transform = transforms.get_transform(
                eval_mode, size, initial_transform=dataset.transform,
                normalize=self.normalize,
                infoprinter=self.infoprinter_, **kwargs
            )
            dataset.transform = transform
            dataset.transforms = vision.StandardTransform(transform=transform)
            dataset.ready_for_use = True

        # GPU and data parallelism
        device, batch_size_per_gpu, nworkers_per_gpu = \
            classifier_toolbox.get_device_batchsize_nworkers(
                self, distributed, gpu, batch_size=batch_size,
                workers=workers, train=False
            )
        if not self.training_in_progress_:
            classifier_toolbox.wrap_and_sendtodevice(
                self, device, infoprinter=self.infoprinter_,
                distributed=distributed, gpu=gpu
            )

        # Evaluation mode
        self.net_.eval()

        if distributed:
            test_sampler = data.distributed.DistributedSampler(
                dataset, shuffle=False, drop_last=True
            )
        else:
            test_sampler = None
        testloader = data.DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=nworkers_per_gpu,
            pin_memory=True, sampler=test_sampler
        )

        with torch.no_grad():
            output, ytrue, batch_nb = self._iterate(
                testloader, eval_mode, return_ref, output_transform, device,
                batch_size, verbose_print_batch
            )

        # Tail data, which have been ignored because of option 'drop_last=True'
        if distributed and (len(testloader.sampler) * world_size < len(testloader.dataset)):
            if compute_tail:
                aux_dataset = data.Subset(
                    testloader.dataset, range(len(testloader.sampler) * world_size,
                    len(testloader.dataset))
                )
                self.infoprinter_("", always=True, allgpus=True)
                self.infoprinter_(
                    "Evaluating tail data ({} elements)".format(len(aux_dataset)),
                    always=True, allgpus=True
                )
                aux_testloader = data.DataLoader(
                    aux_dataset,
                    batch_size=batch_size_per_gpu,
                    shuffle=False,
                    num_workers=nworkers_per_gpu,
                    pin_memory=True
                ) # No sampler. Only the first GPU is working?
                with torch.no_grad():
                    output, ytrue, _ = self._iterate(
                        aux_testloader, eval_mode, return_ref, output_transform,
                        device, batch_size, verbose_print_batch, output=output,
                        ytrue=ytrue, batch_nb=batch_nb
                    )

        self.infoprinter_("", always=True)

        output = torch.cat(output)
        ytrue = torch.cat(ytrue)

        return output, ytrue


    def _iterate(
            self, testloader, eval_mode, return_ref, output_transform,
            device, batch_size, verbose_print_batch, output=None,
            ytrue=None, batch_nb=0
    ):
        # Initialize outputs
        if output is None:
            output = []
        if ytrue is None:
            ytrue = []

        beg_batch_time = time.time()

        for x_batch, y_batch in testloader:
            batch_nb += 1

            # Send to GPU (if available)
            y_batch = y_batch.to(device)
            if isinstance(x_batch, torch.Tensor):
                x_batch = x_batch.to(device)
            elif isinstance(x_batch, list):
                n_tensors = len(x_batch)
                for i in range(n_tensors):
                    x_batch[i] = x_batch[i].to(device)
            else:
                raise ValueError

            if eval_mode == 'shifts':
                if not isinstance(return_ref, bool):
                    raise ValueError(
                        "Please specify a boolean value for 'return_ref'."
                    )
                if return_ref:
                    x_batch, x_batch_ref = x_batch

            if eval_mode in ('tencrops', 'shifts', 'perturbations'): # 5D tensors
                bs, n_crops, nchannels, height, width = x_batch.shape
                # bs may be different from batch_size, when there is not
                # enough remaining images in the dataset.
                x_batch = x_batch.view(
                    -1, nchannels, height, width
                ) # Merge batch size and n_crops

            out = self.net_(x_batch)
            kwargs_transform = {}
            if eval_mode == 'shifts' and return_ref:
                out_ref = self.net_(x_batch_ref)
                kwargs_transform.update(x_ref=out_ref)

            if eval_mode in ('tencrops', 'shifts', 'perturbations'):
                out = out.view(bs, n_crops, *out.shape[1:])

            output_batch = output_transform(out, **kwargs_transform)

            output.append(output_batch)
            ytrue.append(y_batch)
            beg_batch_time, _ = classifier_toolbox.print_time_info(
                beg_batch_time=beg_batch_time, batch_nb=batch_nb,
                verbose_print_batch=verbose_print_batch,
                batch_size=batch_size,
                infoprinter=self.infoprinter_
            )

        return output, ytrue, batch_nb


def _softmax(x, eval_mode):
    out = torch.softmax(x, -1)
    if eval_mode == 'tencrops': # Average over 10 patches
        out = out.mean(1)
    return out


def _compute_kldiv(x, init_transf):

    y = init_transf(x)
    assert len(y.shape) == 3
    n_patches = y.shape[1]

    ref = y[:, :1].repeat(1, n_patches, 1)
    out = torch.sum(ref * torch.log(ref/y), dim=-1)

    return out


def _compute_ypred(x, init_transf):
    y = init_transf(x)
    return torch.argmax(y, dim=-1)
