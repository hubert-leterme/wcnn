import warnings
import time
from datetime import datetime
import math
import torch
from torch.utils import data
from torchvision.datasets import vision

from ... import transforms, toolbox

from .. import classifier_toolbox

# To determine the best validation score, a threshold is applied in order to
# focus on the most significant changes.
THRESHOLD = 1e-4

class TrainerMixin:

    def fit(
            self, dataset, dataset_val=None, save_best_classif=False,
            resume=False, distributed=False, gpu=None, rank=None,
            world_size=None, **kwargs
    ):
        """
        Fit the classifier. The network is trained with stochastic gradient descent.
        When availble, the training is done on parallel GPUs using CUDA interface.

        Parameters
        ----------
        dataset (instance of any class defined in wcnn.data.datasets)
            The training set.
        dataset_val (instance of any class defined in wcnn.data.datasets,
                default=None)
            The validation set. If specified, validation will be performed after
            each training epoch.
        save_best_classif (bool, default=True)
            If set to True, then the classifier is saved as a checkpoint after
            each training epoch. Moreover, a snapshot is kept every p epochs,
            p being specified by the 'snapshots' constructor parameter. Finally,
            if the validation set is specified, then a snapshot of the classifier
            with the highest validation score will be kept.
        resume (bool, default=False)
            If set to True, then training is resumed from an existing partially-
            trained classifier.
        distributed (bool, default=False)
            Whether to train on several GPUs.
        gpu (int, default=None)
            GPU on which data are loaded.
        rank (int, default=None)
            To be provided if the number of nodes is greater than one. If none is
            given, then it is automatically set to the value given by 'gpu'.
        world_size (int, default=None)
            Total number of GPUs (= n_nodes * N_GPUS). If none is given, then it is
            automatically set to N_GPUS.
        save_state_dicts (bool, default=True)
            Whether to save the state dictionaries of the network, optimizer and
            learning rate scheduler.
        commit (str, default=None)
            Hash of the current git commit. Used for the sake of replicability.
        script, args_script (str, default=None)
            Name and arguments of the Python script from which the classifier
            has been created. Used for the sake of replicability.

        Returns
        -------
        self (wcnn.classifier.WaveCnnClassifier)

        """
        if distributed:
            gpu, rank, world_size = classifier_toolbox.get_gpu_rank(
                gpu, rank, world_size
            )

        # Initialize info printer
        self.infoprinter_ = toolbox.InfoPrinter(
            distributed=distributed, gpu=gpu, verbose=self.verbose
        )

        beg_session = time.time()
        beg_elapsed_time = beg_session

        self._check_dataset_integrity(dataset)

        # Transform dataset with compression and conversion to PyTorch tensors
        if not dataset.ready_for_use:
            transform = transforms.get_transform(
                mode='train', size=self.size, initial_transform=dataset.transform,
                normalize=self.normalize,
                data_augmentation=self.data_augmentation, infoprinter=self.infoprinter_
            )
            dataset.transform = transform
            dataset.transforms = vision.StandardTransform(transform=transform)
            dataset.ready_for_use = True

        ########################################################################
        # Case: new classifier, training from scratch
        ########################################################################
        if not resume:

            # Initialize training attributes
            classifier_toolbox.init_train_attr(self, warn=False)

            # Initialize number of classes
            classifier_toolbox.init_n_classes(self, dataset)

            # Initialize network, optimizer and scheduler
            classifier_toolbox.init_network_optimizer_and_scheduler(self)

            # Initialize metrics for validation
            classifier_toolbox.init_metrics(self)

            # Initialize list of gradient distributions,
            # containing as many elements as weight tensors given by
            # 'self.net_.weights_l1reg'.
            if self.get_gradients:
                weights_l1reg = self.net_.weights_l1reg
                self.grads_ = [
                    torch.tensor([]) for _ in weights_l1reg
                ]

            # Number of trainable parameters
            n_params = 0
            for p in self.net_.parameters():
                if p.requires_grad:
                    n_params += p.numel()
            self.n_params_ = n_params

            self.infoprinter_("", always=True)
            self.infoprinter_("Classifier: {}\n".format(self), always=True)
            self.infoprinter_("Network: {}\n".format(self.net_), always=True)
            try:
                pretraining_status = self.net_.pretraining_status
            except AttributeError:
                pass
            else:
                self.infoprinter_(
                    "Pretraining status: {}\n".format(pretraining_status),
                    always=True
                )
            self.infoprinter_(
                "Number of trainable parameters = {}\n".format(self.n_params_),
                always=True
            )

        ########################################################################
        # Case: resuming training from a previously saved classifier
        ########################################################################
        else:
            # Training date and time
            try:
                train_date = self.training_date_and_time_.strftime("%Y-%m-%d")
                train_time = self.training_date_and_time_.strftime("%H:%M:%S")
            except AttributeError as exc:
                raise ValueError(
                    "The classifier hasn't been initialized yet."
                ) from exc

            msg = "Resuming training...\nThe last training session ended on " + \
                "{} at {}.\nNumber of epochs so far = {}"
            msg = msg.format(
                train_date,
                train_time,
                self.current_epoch_
            )
            best_val_err_ = self.best_val_err_ * 100
            msg += "\nLowest validation error achieved so far = {:.2f}%\n"
            msg = msg.format(best_val_err_)
            self.infoprinter_(msg, always=True)

        ########################################################################
        # In any case
        ########################################################################

        self.training_in_progress_ = True

        newtime = datetime.now()
        msg = "New training session started on {} at {}.\n"
        msg = msg.format(
            newtime.strftime("%Y-%m-%d"), newtime.strftime("%H:%M:%S")
        )
        self.infoprinter_(msg, always=True)

        if self.accumulate_grads_insteadof_averaging:
            warnings.warn(
                "WARNING: accumulating gradient instead of averaging. "
                "This may not be an appropriate behavior."
            )

        # GPU and data parallelism
        device, batch_size_train_per_gpu, nworkers_per_gpu = \
            classifier_toolbox.get_device_batchsize_nworkers(
                self, distributed, gpu
            )
        self.infoprinter_("(Sub-)Batch size per GPU = {}".format(
            batch_size_train_per_gpu
        ))
        self.infoprinter_("Number of workers:", nworkers_per_gpu)
        self.device_train_ = device
        classifier_toolbox.wrap_and_sendtodevice(
            self, device, infoprinter=self.infoprinter_,
            distributed=distributed, gpu=gpu
        )

        if distributed:
            self.infoprinter_("Creating training data sampler...")
            train_sampler = data.distributed.DistributedSampler(
                dataset, shuffle=self.shuffle_data
            )
            self.infoprinter_("...done.")
        else:
            train_sampler = None
        self.infoprinter_("Creating training data loader...")
        trainloader = data.DataLoader(
            dataset,
            batch_size=batch_size_train_per_gpu,
            shuffle=(train_sampler is None and self.shuffle_data),
            num_workers=nworkers_per_gpu,
            pin_memory=True, sampler=train_sampler
        )
        self.infoprinter_("...done.")

        # Define loss and metrics for evaluation on a validation set
        # Cross-entropy loss. The loss function takes as arguments a tensor
        # of shape (K, num_classes) (output of the network) and a tensor of
        # shape (K,) (ground-truth vector), where K is the number of training
        # exemples in a batch.
        self.infoprinter_("Initializing loss...")
        loss_fun = _L1RegCrossEntropyLoss(
            has_l1loss=self.has_l1loss, lambda_params=self.lambda_params
        ).to(device)
        self.infoprinter_("...done.")

        # Keyword arguments to compute the loss
        kwargs_loss = {}
        if self.has_l1loss:
            kwargs_loss.update(weights=self.net_.weights_l1reg)

        # List of gradients
        if self.get_gradients:
            grads = [[] for _ in self.net_.weights_l1reg]

        # TODO Compute validation scores before starting training

        while self.current_epoch_ < self.epochs:

            beg_epoch = time.time()
            self.net_.train() # Train mode (must be reset because validation
                              # is performed after each epoch)
            self.current_epoch_ += 1
            self.infoprinter_("Passing current epoch to the sampler...")
            if distributed:
                train_sampler.set_epoch(self.current_epoch_)
            self.infoprinter_("...done.")

            # Information about current epoch and learning rate
            current_lr = self.optimizer_.param_groups[0]['lr']
            self.lr_history_[self.current_epoch_] = current_lr
            self.infoprinter_("Epoch {}, learning rate = {:.1e}:\n".format(
                self.current_epoch_,
                current_lr
            ), always=True)

            # Training for one epoch
            beg_batch_time = time.time()
            batch_nb = 0
            running_loss = 0.0
            accum_track = 0
            accum_loss = 0.0
            self.optimizer_.zero_grad()
            for x_batch, y_batch in trainloader:
                accum_track += 1
                self.infoprinter_("Batch size = ", x_batch.shape[0])

                # Send to GPU (if available)
                self.infoprinter_("Sending batches to device...")
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                self.infoprinter_("...done.")

                # Forward-propagation
                self.infoprinter_("Forward propagation...")
                outputs = self.net_(x_batch)
                self.infoprinter_("...done.")

                # Backward-propagation and weight update using SGD

                ## Loss function
                self.infoprinter_("Computing loss...")
                loss = loss_fun(outputs, y_batch, **kwargs_loss)
                self.infoprinter_("...done.")

                # Backward-propagation. The loss is divided by the number
                # of batch groups because the gradients are accumulated at
                # each call of backward().
                self.infoprinter_("Backward propagation...")
                if not self.accumulate_grads_insteadof_averaging:
                    loss = loss / self.batch_accum
                accum_loss += loss.item()
                loss.backward()
                self.infoprinter_("...done.")

                if accum_track == self.batch_accum:
                    batch_nb += 1

                    # Print statistics
                    self.losses_.append(accum_loss)
                    running_loss += accum_loss
                    beg_batch_time, running_loss = \
                        classifier_toolbox.print_time_info(
                            beg_batch_time=beg_batch_time, batch_nb=batch_nb,
                            verbose_print_batch=self.verbose_print_batch,
                            batch_size=self.batch_size_train, infoprinter=self.infoprinter_,
                            running_loss=running_loss
                        ) # Reset beg_batch_time and running_loss after each effective printing

                    # Register gradients
                    if self.get_gradients:
                        for j, wght in enumerate(self.net_.weights_l1reg):
                            # For the j-th weight tensor, get the l-infty norm of
                            # the gradient with respect to each row vector (one value
                            # per output channel)
                            new_grad = torch.linalg.norm(
                                wght.grad, ord=math.inf, dim=1
                            ).flatten()
                            grads[j].append(new_grad)

                    # Update the weights and reset gradients
                    self.infoprinter_("Performing gradient descent step...")
                    self.optimizer_.step()
                    self.infoprinter_("...done.")

                    # Reset gradients and accumulated metrics
                    self.infoprinter_("Gradient reset...")
                    self.optimizer_.zero_grad()
                    accum_track = 0
                    accum_loss = 0.0
                    self.infoprinter_("...done.")


            # Register gradients
            if self.get_gradients:
                for j, (vals, vals_epoch) in enumerate(zip(
                        self.grads_, grads
                )):
                    self.grads_[j] = torch.cat([vals, *vals_epoch])
                    vals_epoch.clear() # Clear list of gradients


            # Validate and check best classif, if required
            self.infoprinter_("", always=True)
            self.infoprinter_("Validation", always=True)
            self.infoprinter_("----------", always=True)
            err0, best_classif = self._validate_and_check_best_classif(
                dataset_val, distributed=distributed, gpu=gpu, rank=rank,
                world_size=world_size
            )

            # Update learning rate
            self._update_learning_rate(err0)

            # Save classifier and print time (first GPU only)
            if (not distributed) or (rank == 0):
                self._save_and_print_time(
                    best_classif, save_best_classif, beg_elapsed_time, beg_session,
                    beg_epoch, load_grads=self.get_gradients,
                    **kwargs
                )

        # End of training
        self.infoprinter_("Training stopped after {} epochs.".format(
            self.current_epoch_
        ), always=True)
        self.elapsed_time_ += time.time() - beg_elapsed_time
        self.training_in_progress_ = False

        return self


    def _validate_and_check_best_classif(
            self, dataset_val, **kwargs
    ):
        # Validation if required
        if dataset_val is not None:
            beg_validation = time.time()

            # Validate
            val_score = self.score(dataset_val, save_results=False, **kwargs)
            val_err = toolbox.get_error(val_score)

            # Record score and time
            self.val_errs_[self.current_epoch_] = val_err
            self.val_times_[self.current_epoch_] = time.time() - beg_validation

            # Print information
            if isinstance(val_err, float):
                err0 = val_err
                msg = "\n==> Validation error = {:.2f}".format(val_err)

            elif isinstance(val_err, dict):
                err0 = val_err[1] # top-1 accuracy score
                msg = "\n==> Validation errors: "
                for nth in val_err.keys():
                    msg += "top-{} = {:.2f}; ".format(nth, val_err[nth])
                msg = msg[:-2]

            # Check if the current state outperforms the previous ones
            if err0 < self.best_val_err_ - THRESHOLD:
                self.best_val_err_ = err0
                msg += " (the best so far)"
                best_classif = True
            else:
                best_classif = False

            # Print information
            self.infoprinter_(msg, always=True)

        else:
            err0 = None
            best_classif = None

        return err0, best_classif


    def _save_and_print_time(
            self, best_classif, save_best_classif, beg_elapsed_time, beg_session,
            beg_epoch, **kwargs
    ):
        # Training date and time
        date_and_time = datetime.now()
        self.date_and_time_per_epoch_[self.current_epoch_] = date_and_time
        self.training_date_and_time_ = date_and_time

        # Elapsed time
        elapsed_time = self.elapsed_time_ + time.time() - beg_elapsed_time
        elapsed_time_for_current_session = time.time() - beg_session
        elapsed_time_for_current_epoch = time.time() - beg_epoch

        self.elapsed_time_per_epoch_[self.current_epoch_] = elapsed_time_for_current_epoch

        # Save the current state if it outperforms the previous ones (overwrites)
        if best_classif and save_best_classif:
            self.save(status='best', elapsed_time_=elapsed_time, **kwargs)

        # Checkpoint (overwrites the previous checkpoint)
        kwargs_checkpoint = dict(**kwargs)
        self.save(elapsed_time_=elapsed_time, **kwargs_checkpoint)

        # Snapshot (saved in a new file)
        if self.snapshots > 0 and self.current_epoch_ % self.snapshots == 0:
            self.save(status='snapshot', elapsed_time_=elapsed_time, **kwargs)

        # Print information
        self.infoprinter_("\nElapsed time = {:.0f}:{:.0f}:{:.0f}".format(
            elapsed_time // 3600,
            (elapsed_time % 3600) // 60,
            elapsed_time % 60
        ), always=True)
        self.infoprinter_("Elapsed time for the current session = {:.0f}:{:.0f}:{:.0f}".format(
            elapsed_time_for_current_session // 3600,
            (elapsed_time_for_current_session % 3600) // 60,
            elapsed_time_for_current_session % 60
        ), always=True)
        msg = "Elapsed time for the current epoch = {:.0f}:{:.0f}:{:.0f} "
        msg = msg.format(
            elapsed_time_for_current_epoch // 3600,
            (elapsed_time_for_current_epoch % 3600) // 60,
            elapsed_time_for_current_epoch % 60
        )
        try:
            validation_time = self.val_times_[self.current_epoch_]
            msg += "(incl. {:.0f}:{:.0f}:{:.0f} for validation)\n"
            msg = msg.format(
                validation_time // 3600,
                (validation_time % 3600) // 60,
                validation_time % 60
            )
        except KeyError:
            pass
        msg += "\n"

        self.infoprinter_(msg, always=True)


    def _update_learning_rate(self, err0):

        if isinstance(
                self.lr_scheduler_,
                torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.lr_scheduler_.step(err0)

        elif self.lr_scheduler_ is not None:
            self.lr_scheduler_.step()


class _L1RegCrossEntropyLoss(torch.nn.CrossEntropyLoss):

    def __init__(self, *args, has_l1loss=False, lambda_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_l1loss = has_l1loss
        self._lambda_params = lambda_params

    def forward(self, inp, target, weights=None):
        """
        Paramters
        ---------
        inp (torch.Tensor)
            Tensor of shape (n_imgs, n_classes).
        target (torch.Tensor)
            Tensor of shape (n_imgs), containing class indices.
        weights (list of torch.Tensor, default=None)
            Tensors of shape (out_channels, in_channels_per_group, 1, ..., 1).
            List of weights for 1x1 convolution layers.

        """
        if self._has_l1loss:
            penalization = self._l1loss(weights, self._lambda_params)
        else:
            penalization = 0.
        return super().forward(inp, target) + penalization

    def _l1loss(self, weights, lambda_params):
        out = 0
        for wght, lbd in zip(weights, lambda_params):
            # Vectors of norm-1 and norm-inf, size = out_channels
            norm1 = torch.linalg.norm(wght, ord=1, dim=1).flatten()
            norminf = torch.linalg.norm(wght, ord=math.inf, dim=1).flatten()
            out += lbd * (torch.sum(norm1/norminf - 1))
        return out
