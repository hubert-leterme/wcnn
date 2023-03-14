import copy
from os.path import join
import warnings

import torch
from torch import nn

from ... import toolbox

from .. import classifier_toolbox

LIST_OF_KWARGS_DB = ['commit', 'script', 'args_script']
LIST_OF_KWARGS_INITNET = ['use_old_confignames']
LIST_OF_KWARGS_LOADSTATEDICT = ['backward_compatibility', 'modulus']

class SaverLoaderMixin:

    def save(self, status='checkpoint', save_state_dicts=True,
             path_to_classif=None, path_to_csv=None, **kwargs):
        """
        Pickle the classifier and save the PyTorch state dictionaries (newtork,
        optimizer and learning rate scheduler) in a separate file using torch.save.

        Parameters
        ----------
        status (str, default='checkpoint')
            Can be one of 'checkpoint' (last checkpoint; will overwrite the
            previous one), 'snapshot' (keep a snapshot of the classifier at a
            specific training epoch) or 'best' (classifier with the best
            validation score; will overwrite the previous one). This parameter
            will have an impact on the filename.
        save_state_dicts (bool, default=True)
            Whether to save the state dictionaries of the network, optimizer and
            learning rate scheduler.
        path_to_classif (str, default=None)
            Directory where to save the classifier. If none is given, a config file
            'wcnn_config.yml' must be provided.
        path_to_csv (str, default=None)
            Directory containing the CSV file (reference database for all
            classifiers). If none is given, a config file 'wcnn_config.yml' must be
            provided.
        commit (str, default=None)
            Hash of the current git commit. Used for the sake of replicability.
        script, args_script (str, default=None)
            Name and arguments of the Python script from which the classifier
            has been created. Used for the sake of replicability.
        **kwargs
            Attributes to update before saving

        """
        # Set training_in_progress_ flag to False
        training_in_progress = self.training_in_progress_
        self.training_in_progress_ = False

        # Remove network, optimizer and learning rate scheduler (stored in a
        # separate file)
        if self.grads_ is not None:
            net, optimizer, lr_scheduler, grads = self.unload_net('grads_')
        else:
            net, optimizer, lr_scheduler = self.unload_net()
            grads = None

        # Path
        path_to_classif = classifier_toolbox.set_path_classif(path_to_classif)

        # Load classifier reference database and check if a file corresponding
        # to the given name and status already exists
        if status not in ['checkpoint', 'snapshot', 'best']:
            raise NotImplementedError
        if status == 'snapshot':
            status = 'e{}'.format(self.current_epoch_) # E.g., 'e100'

        kwargs_db = toolbox.extract_dict(LIST_OF_KWARGS_DB, kwargs)

        filename = classifier_toolbox.get_filename_and_update_db(
            index_db=(self.name, status),
            path_to_csv=path_to_csv, **kwargs_db
        )

        # Create shallow copy of the classifier
        classif = copy.copy(self)

        filename = join(path_to_classif, filename)

        if save_state_dicts:

            # Unwrap network
            unwrapped_net = net
            while isinstance(unwrapped_net, nn.parallel.DistributedDataParallel):
                unwrapped_net = unwrapped_net.module

            # Save the PyTorch objects into a separate file
            state_dicts = dict(
                net=unwrapped_net.state_dict(),
                optimizer=optimizer.state_dict()
            )
            if lr_scheduler is not None:
                state_dicts.update(
                    lr_scheduler=lr_scheduler.state_dict()
                )
            torch.save(state_dicts, filename + '.tar')
            if grads is not None:
                torch.save(grads, filename + '.grads')

        # Update attributes (in the copy only)
        for param in kwargs:
            setattr(classif, param, kwargs[param])
        torch.save(classif, filename + '.classif')

        # Set training_in_progress_ flag in its previous state
        self.training_in_progress_ = training_in_progress

        # Plug back the PyTorch objects into the classifier
        self.net_ = net
        self.optimizer_ = optimizer
        self.lr_scheduler_ = lr_scheduler
        if grads is not None:
            self.grads_ = grads


    @staticmethod
    def load(
            id_classif, status='checkpoint', path_to_classif=None,
            path_to_csv=None, load_net=True, warn=True,
            load_grads=False, train=True, **kwargs
    ):
        """
        Load classifier and update state_dict from PyTorch objects.

        Parameters
        ----------
        id_classif (str)
            Name of the classifier.
        status (str, default='checkpoint')
            One of 'checkpoint', 'best' or 'e<epochs>'.
        path_to_classif (str, default=None)
            Directory where to load the classifier. If none is given, a config file
            'wcnn_config.yml' must be provided.
        path_to_csv (str, default=None)
            Directory containing the CSV file (reference database for all
            classifiers). If none is given, a config file 'wcnn_config.yml' must be
            provided.
        load_net (bool, default=True)
            Whether to load the state dictionaries into the classifier (see
            'load_net' method). Can be set to False if no training or evaluation
            is required.
        load_grads (bool, default=False)
            Whether to load learning attribute 'self.grads_'. Only used if
            'load_net' is set to True.
        train (bool, default=True)
            Whether to set the network in training or evaluation mode.
        warn (bool, default=True)
        download (bool, default=False)
            Whether to download the classifier from a remote SSH host.
        hostname (str, default=None)
            Name of the SSH host. Must be provided if download=True.
        download_statedicts (bool, default=False)
            Whether to download the PyTorch state dictionaries (i.e., the *.tar and
            *.grads files) in addition to the main pickled object containing the
            training and evaluation metrics (i.e., the *.classif file).
        backward_compatibility (bool, default=False)
            Argument for method 'load_state_dict'.
        modulus (bool, default=False)
            Whether the network implements modulus instead of max pooling. Only
            valid if 'backward_compatibility' is set to True.
        ignore_attr_consistency (bool, default=False)
            Whether to check whether the classifier's attributes match the keyword
            arguments provided by **kwargs. Also check whether the current commit
            number matches the one provided when first saving the checkpoint.
        commit (str, default=None)
            See 'ignore_attr_consistency'
        **kwargs
            See 'ignore_attr_consistency'

        Returns
        ----------
        classif (wcnn.classifier.WaveCnnClassifier)

        """
        path_to_classif = classifier_toolbox.set_path_classif(path_to_classif)

        kwargs_init = toolbox.extract_dict(
            LIST_OF_KWARGS_INITNET, kwargs
        )
        kwargs_loadstatedict = toolbox.extract_dict(
            LIST_OF_KWARGS_LOADSTATEDICT, kwargs
        )

        classif = classifier_toolbox.load_classif(
            index_db=(id_classif, status),
            path_to_csv=path_to_csv, path_to_classif=path_to_classif,
            warn=warn, **kwargs
        )

        # In case the name has been changed
        if classif.name != id_classif:
            if warn:
                msg = "The classifier's name has changed: previous name = {}; new name = {}."
                msg = msg.format(classif.name, id_classif)
                warnings.warn(msg)
            classif.name = id_classif

        # For backward compatibility
        classifier_toolbox.init_train_attr(classif, warn=warn)
        classifier_toolbox.init_metrics(classif)

        # Load network, optimizer and learning rate scheduler
        if load_net:
            classif.load_net(
                status=status, path_to_classif=path_to_classif,
                path_to_csv=path_to_csv, load_grads=load_grads,
                train=train, **kwargs_init, **kwargs_loadstatedict
            )

        return classif


    @staticmethod
    def list(index=None):
        out, _ = classifier_toolbox.load_db()
        if index is not None:
            out = out.loc[index]
        return out


    def load_net(
            self, status, path_to_classif=None,
            path_to_csv=None, load_grads=False, train=True, **kwargs
    ):
        """
        Initialize the network, optimizer and learning rate scheduler and load
        the state dictionaries.

        Parameters
        ----------
        status (str)
            One of 'checkpoint', 'best' or 'e<epochs>'.
        train (bool, default=True)
            Whether to set the network in training or evaluation mode.
        path_to_classif (str, default=None)
            Directory where to save the classifier. If none is given, a config file
            'wcnn_config.yml' must be provided.
        path_to_csv (str, default=None)
            Directory containing the CSV file (reference database for all
            classifiers). If none is given, a config file 'wcnn_config.yml' must be
            provided.
        load_grads (bool, default=False)
            Whether to load learning attribute 'self.grads_'.
        backward_compatibility (bool, default=False)
            Argument for method 'load_state_dict'.
        modulus (bool, default=False)
            Whether the network implements modulus instead of max pooling. Only
            valid if 'backward_compatibility' is set to True.

        """
        kwargs_init = toolbox.extract_dict(
            LIST_OF_KWARGS_INITNET, kwargs
        )
        kwargs_loadstatedict = toolbox.extract_dict(
            LIST_OF_KWARGS_LOADSTATEDICT, kwargs
        )
        # Initialize PyTorch objects. Argument 'skip_pretrained_freeconv'
        # is set to True because the trainable parameters will be
        # updated with the loaded state dictionary (see below)
        classifier_toolbox.init_network_optimizer_and_scheduler(
            self, train=train, skip_pretrained_freeconv=True, **kwargs_init
        )

        path_to_classif = classifier_toolbox.set_path_classif(path_to_classif)
        path_to_csv = classifier_toolbox.set_path_csv(path_to_csv)

        # Load classifier reference database
        df, _ = classifier_toolbox.load_db(path_to_csv=path_to_csv)
        dbentry = df.loc[(self.name, status)]
        filename = dbentry.filename

        # Load saved states
        path = join(path_to_classif, filename + '.tar')
        try:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        except FileNotFoundError:
            warnings.warn(
                "Model state dicts not found; initialized from scratch."
            )
        else:
            self.net_.load_state_dict(
                checkpoint['net'], **kwargs_loadstatedict
            )
            if train:
                self.optimizer_.load_state_dict(checkpoint['optimizer'])
                if self.lr_scheduler_ is not None:
                    self.lr_scheduler_.load_state_dict(checkpoint['lr_scheduler'])
            if load_grads:
                path = join(path_to_classif, filename + '.grads')
                grads = torch.load(path, map_location=torch.device('cpu'))
                self.grads_ = grads


    def unload_net(self, *attrs):
        """
        Unload network, optimizer and learning rate scheduler and return these
        objects.

        Paramters
        ---------
        *attrs (str)
            Attributes to remove from the classifier.

        """
        return toolbox.pop_attrs(
            self, 'net_', 'optimizer_', 'lr_scheduler_', *attrs
        )
