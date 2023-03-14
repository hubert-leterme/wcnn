import os
import uuid
import io
import sys
import inspect
import time
from datetime import datetime
import warnings
import pickle
try:
    import paramiko # SSH connection
except ImportError:
    pass
import scp

import pandas as pd
import torch
from torch import optim
import nvidia_smi

from .. import CUDA, DEVICE, N_GPUS, CONFIGDICT
from .. import toolbox
from ..cnn import models, base_modules, cnn_toolbox
from .. import metrics as metr

ONE_MIB = 1024**2
OLDATTRS = {
    "net": "arch",
    "dataloader_nworkers": "workers",
    "n_epochs": "epochs"
} # Backward compatibility

# Dictionary of attributes to be set during training.
# For each attribute, the constructor is provided. Calling this constructor
# with no argument yields the default value.

def _default_datetime():
    return datetime(1, 1, 1)

def _infty():
    return float('inf')

TRAIN_ATTRS = dict(
    elapsed_time_=int,
    elapsed_time_per_epoch_=dict,
    n_classes_=None,
    n_params_=None,
    losses_=list,
    grads_=list,
    current_epoch_=int,
    lr_history_=dict,
    training_in_progress_=bool,
    device_train_=None,
    training_date_and_time_=_default_datetime,
    date_and_time_per_epoch_=dict,
    net_=None,
    optimizer_=None,
    lr_scheduler_=None,
    metrics_=None,
    args_metrics_=dict,
    val_errs_=dict,
    val_times_=dict,
    remove_epochs_=list,
    best_val_err_=_infty, # Smallest validation error
                          # (initialized to infinity)
    infoprinter_ = None,
    eval_scores_=dict
)

# See docstring wcnn.classifier for a description of LR_SCHEDULER_DICT
LR_SCHEDULER_DICT = {
    'StepLR25': (optim.lr_scheduler.StepLR, dict(step_size=25)),
    'StepLR30': (optim.lr_scheduler.StepLR, dict(step_size=30)),
    'StepLR100': (optim.lr_scheduler.StepLR, dict(step_size=100)),
    'ReduceLROnPlateau': (optim.lr_scheduler.ReduceLROnPlateau, {}),
    None: (None, None)
}

# Substitute attribute values (backward compatibility)
ATTR_SUBSTITUTION = dict(arch={})
for nlayers in (18, 34, 50, 101):
    ATTR_SUBSTITUTION["arch"].update(**{
        "ResNet{}".format(nlayers): "resnet{}".format(nlayers),
        "ZhangResNet{}".format(nlayers): "zhang_resnet{}".format(nlayers),
        "ZouResNet{}".format(nlayers): "zou_resnet{}".format(nlayers),
        "DtCwptResNet{}".format(nlayers): "dtcwpt_resnet{}".format(nlayers),
        "DtCwptZhangResNet{}".format(nlayers): "dtcwpt_zhang_resnet{}".format(nlayers),
        "DtCwptZouResNet{}".format(nlayers): "dtcwpt_zou_resnet{}".format(nlayers)
    })

# List of keyword arguments to be passed as network's constructor parameters
LIST_OF_KWARGS_NET = [
    'config', 'wavelet', 'backend', 'trainable_qmf', 'padding_mode',
    'pretrained_model', 'status_pretrained'
]

LIST_OF_KWARGS_LOAD = ['path_to_classif', 'path_to_csv']

LIST_OF_KWARGS_FORWARDEVAL = [
    'shift_direction', 'max_shift', 'pixel_divide', 'batch_size',
    'verbose_print_batch', 'size', 'workers',
    'downsampling_factor', 'return_ref'
]

class _RenameUnpickler(pickle.Unpickler):
    """
    Used to handle ModuleNotFoundError after renaming 'wavenet' to 'wcnn'.

    """
    def find_class(self, module, name):
        splitmodules = module.split('.')
        if splitmodules[0] == 'wavenet':
            splitmodules[0] = 'wcnn'
            module = '.'.join(splitmodules)

        return super().find_class(module, name)

pickle.Unpickler = _RenameUnpickler # Monkey-patching pickle


def init_train_attr(classif, train_attrs=None, warn=True):
    """
    Initialize attributes before training (except constructor's parameters).
    This method is also called after loading an existing classifier, for
    backward compatibility.
    See https://scikit-learn.org/stable/developers/develop.html for more info.

    """
    if train_attrs is None:
        train_attrs = TRAIN_ATTRS

    # In any case
    for attr in train_attrs:
        if attr not in classif.__dict__:
            if train_attrs[attr] is not None:
                default_val = train_attrs[attr]()
            else:
                default_val = None
            setattr(classif, attr, default_val)
            if warn:
                msg = "Attribute '{}' not found; initialized to {}.".format(
                    attr, default_val
                )
                warnings.warn(msg)

    return classif


def _update_attr_backward_compatibility(classif, dict_attrval, new_attr,
                                        old_attr=None, args_old_attr=None):
    if old_attr is None:
        old_attr = new_attr # Attribute name didn't change
    if args_old_attr is None:
        args_old_attr = 'args_' + new_attr
    for attrval in dict_attrval:
        old_attrval = dict_attrval.get(attrval)[0]
        args_old_attrval = dict_attrval.get(attrval)[1]
        if getattr(classif, old_attr) == old_attrval \
                and getattr(classif, args_old_attr) == args_old_attrval:
            setattr(classif, new_attr, attrval)


def init_n_classes(classif, dataset):
    classif.n_classes_ = dataset.num_classes
    return classif


def init_network_optimizer_and_scheduler(classif, train=True, **kwargs):
    """
    Parameters
    ----------
    train (bool, default=True)
        Whether to set the network in training or evaluation mode.

    """
    net_class = models.__dict__[classif.arch]
    lr_sch_class, kwargs_lr_sch = LR_SCHEDULER_DICT[classif.lr_scheduler]

    params = classif.get_params().copy() # Scikit-learn built-in method

    # Get keyword arguments for the network
    kwargs_net = toolbox.extract_dict(
        LIST_OF_KWARGS_NET, params, discard_none=True
    )

    classif.net_ = net_class(
        num_classes=classif.n_classes_, **kwargs_net, **kwargs
    )
    if not train:
        classif.net_.eval()

    classif.optimizer_ = optim.SGD(
        classif.net_.parameters(),
        lr=classif.lr,
        momentum=classif.momentum,
        weight_decay=classif.weight_decay
    )
    if lr_sch_class is not None:
        classif.lr_scheduler_ = lr_sch_class(
            classif.optimizer_, **kwargs_lr_sch
        )

    return classif


def init_metrics(classif):

    if classif.metrics is None:
        classif.metrics = 'top1_accuracy_score'

    classif.metrics_, classif.args_metrics_ = \
        metr.METRICS_DICT[classif.metrics]

    return classif


def get_device_batchsize_nworkers(
        classif, distributed, gpu=None, batch_size=None,
        workers=None, train=True
):
    if batch_size is None:
        if train:
            if classif.batch_size_train % classif.batch_accum:
                msg = (
                    "The training batch size ({}) must be divisible "
                    "by attribute 'batch_accum' ({})."
                )
                raise ValueError(msg.format(
                    classif.batch_size_train, classif.batch_accum
                ))
            batch_size = classif.batch_size_train // classif.batch_accum
        else:
            batch_size = classif.batch_size_test

    if workers is None:
        workers = classif.workers
    if workers is None:
        try:
            workers = len(os.sched_getaffinity(0))
        except Exception:
            workers = 0

    if distributed and CUDA:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if gpu is not None:
            device = torch.device("cuda:{}".format(gpu))
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            if batch_size % N_GPUS != 0:
                msg = (
                    "The batch size ({}) must be divisible by the number "
                    "of GPUs ({})."
                )
                raise ValueError(msg.format(batch_size, N_GPUS))
            batch_size_per_gpu = batch_size // N_GPUS
            nworkers_per_gpu = (workers + N_GPUS - 1) // N_GPUS
        else:
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            device = DEVICE
            batch_size_per_gpu = batch_size
            nworkers_per_gpu = workers
    else:
        device = DEVICE
        batch_size_per_gpu = batch_size
        nworkers_per_gpu = workers

    return device, batch_size_per_gpu, nworkers_per_gpu


def wrap_and_sendtodevice(
        classif, device, infoprinter, distributed=False, gpu=None
):
    """
    Send to GPU if available and parallelize forward-propagation.

    """
    if CUDA:
        infoprinter("Using", N_GPUS, "GPUs\n", always=True)

        nvidia_smi.nvmlInit()
        if gpu is None:
            gpu = 0
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        infoprinter("GPU total memory: {:.2f} MiB".format(
            info.total / ONE_MIB
        ))
        infoprinter("GPU free memory: {:.2f} MiB".format(
            info.free / ONE_MIB
        ))
        infoprinter("GPU used memory: {:.2f} MiB".format(
            info.used / ONE_MIB
        ))
        nvidia_smi.nvmlShutdown()

    infoprinter("Model size: {:.2f} MiB".format(
        cnn_toolbox.get_model_size(classif.net_) / ONE_MIB
    ))

    # Send network parameters and gradients to GPU
    infoprinter("Send network to device...")
    classif.net_.to(device)
    infoprinter("...done.")
    if classif.grads_ is not None:
        infoprinter("Send gradient info to device...")
        for i, grd in enumerate(classif.grads_):
            classif.grads_[i] = grd.to(device)
        infoprinter("...done.")

    # Send optimizer parameters to GPU (e.g. momentum)
    if classif.optimizer_ is not None:
        infoprinter("Send optimizer to device...")
        for state in classif.optimizer_.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        infoprinter("...done.")
    else:
        warnings.warn("Optimizer not found.")

    # Wrap network into DistributedDataParallel
    if distributed:
        infoprinter("Wrap network into DistributedDataParallel...")
        classif.net_ = base_modules.DistributedDataParallel(
            classif.net_, device_ids=[gpu]
        )
        infoprinter("...done.")

    return classif


def print_time_info(
        beg_batch_time, batch_nb, verbose_print_batch, batch_size,
        infoprinter, running_loss=None
):
    if batch_nb % verbose_print_batch == 0:
        inf = batch_size * (batch_nb - verbose_print_batch) + 1
        sup = batch_size * batch_nb
        msg = "\tData {}-{}".format(inf, sup)
        # Running loss
        if running_loss is not None:
            msg += "\t\t{} = {:.2f}".format(
                "Running loss", running_loss / verbose_print_batch
            )
            running_loss = 0.0
        msg += "\t\tTime = {:.0f} secs".format(time.time() - beg_batch_time)
        infoprinter(msg, always=True)
        beg_batch_time = time.time()

    return beg_batch_time, running_loss


def get_gpu_rank(gpu, rank, world_size):
    if gpu is None:
        gpu = 0
    if rank is None:
        rank = gpu
    if world_size is None:
        world_size = N_GPUS
    return gpu, rank, world_size


def load_classif(
        index_db, path_to_csv=None, path_to_classif=None,
        download=False, hostname=None,
        download_statedicts=False, warn=None, exclude=None,
        ignore_attr_consistency=False, commit=None, **kwargs
):
    dbentry = _get_dbentry(
        index_db, path_to_csv=path_to_csv, path_to_classif=path_to_classif
    )
    filename = dbentry.filename

    saved_commit = dbentry.commit
    script = dbentry.script
    args_script = dbentry.args_script
    if isinstance(filename, pd.core.series.Series):
        if filename.size > 1:
            raise ValueError("More than one file corresponds to this index.")
        filename = filename[0]
        saved_commit = saved_commit[0]
        script = script[0]
        args_script = args_script[0]

    path = '.'.join([filename, "classif"]) # Filename with extension
    path = os.path.join(path_to_classif, path) # Complete path

    # Download object upon request
    if download:
        download_classif(
            index_db=index_db, hostname=hostname,
            path_to_csv=path_to_csv, path_to_classif=path_to_classif,
            download_statedicts=download_statedicts
        )

    # Load object
    try:
        try:
            classif = torch.load(
                path, map_location=torch.device('cpu'), pickle_module=pickle
            ) # Use the monkey-patched pickle module
        except AttributeError:
            # Occurs if the classifier was saved with pickle instead of torch.save.
            # Would raise RuntimeError("Invalid magic number; corrupt file?") if the
            # classifier was not confused with an integer.
            # See https://discuss.pytorch.org/t/runtime-error-when-loading-pytorch-model-from-pkl/38330
            with open(path, 'rb') as fo:
                classif = _RenameUnpickler(fo).load()
    except FileNotFoundError as exc:
        msg = "The classifier could not be found. Consider downloading it from the " + \
              "cluster where it has been created (filename without extension = " + \
              "'{}'), or run again the script with " + \
              "the following settings:\n" + \
              "Commit number = {}\n" + \
              "Script = {}\n" + \
              "Arguments = {}\n\n" + \
              "You can retrieve the original conda environment with the config " + \
              "file 'condaenv.yml' at the root of the git repository."
        msg = msg.format(filename, saved_commit, script, args_script)
        raise ClassifNotFoundError(msg) from exc

    # Check missing attributes (for backward compatibility)
    kwargs_check = {}
    if warn is not None:
        kwargs_check.update(warn=warn)
    if exclude is not None:
        kwargs_check.update(exclude=exclude)
    _check_consistency_and_repair(classif, **kwargs_check)

    # Update classifier
    updatable_attrs = [
        'workers', 'epochs', 'verbose_print_batch', 'verbose',
        'snapshots'
    ]
    kwargs_update = toolbox.extract_dict(updatable_attrs, kwargs)
    _update_attrs(classif, **kwargs_update)

    # Check consistency on other attributes, if requested
    if saved_commit is not None:
        if commit is not None and commit != saved_commit:
            msg = "Wrong commit number.\n"
            msg += "Expected: {};\n".format(saved_commit)
            msg += "Provided: {}.\n".format(commit)
            if not ignore_attr_consistency:
                raise ValueError(msg)
            warnings.warn(msg)
    for attr_name, attr in kwargs.items():
        expected_attr = getattr(classif, attr_name)
        if expected_attr != attr:
            msg = (
                "A classifier corresponding to the specified index already exists "
                "in the database but is inconsistent with the provided arguments. "
                "'{}' was expected to be equal to '{}', not '{}'."
            )
            msg = msg.format(attr_name, expected_attr, attr)
            if not ignore_attr_consistency:
                raise ValueError(msg)
            warnings.warn(msg)

    return classif


def _update_attrs(classif, **kwargs):

    for attr_name, new_attr in kwargs.items():
        old_attr = getattr(classif, attr_name)
        if new_attr != old_attr:
            warnings.warn(
                "Attribute '{}' changed: previous value = {}; new value = {}.".format(
                    attr_name, old_attr, new_attr
                )
            )
            setattr(classif, attr_name, new_attr)


def download_classif(
        index_db, hostname, path_to_csv=None,
        path_to_classif=None, remotepath=None, download_statedicts=False
):
    class _ProgressBar:
        # Define progress callback that prints the current percentage completed for the file.
        # Adapted from https://pypi.org/project/scp/
        def __init__(self):
            self._reset()

        def _reset(self):
            self.start = False
            self.strlength = 0
            self.threshold = 0.

        def __call__(self, filename, size, sent):
            if not self.start:
                sys.stdout.write(
                    "Downloading {}: ".format(filename)
                )
                sys.stdout.flush()
                self.start = True

            progress = sent / size * 100
            if progress >= self.threshold:
                for _ in range(self.strlength):
                    sys.stdout.write('\b')
                    sys.stdout.flush()
                progress = "{:.0f} %".format(progress)
                sys.stdout.write(progress)
                self.strlength = len(progress) # Erase at next call
                self.threshold += 10.

        def newline(self):
            sys.stdout.write('\n')
            self._reset()

    if remotepath is None:
        if CONFIGDICT is not None:
            remotepath = CONFIGDICT["gpu_clusters"].get(hostname)
        if remotepath is None:
            raise ValueError("Missing value for argument 'remotepath'.")
    remotepath = os.path.expanduser(remotepath)

    dbentry = _get_dbentry(
        index_db, path_to_csv=path_to_csv, path_to_classif=path_to_classif
    )
    filename = dbentry.filename

    filename = os.path.join(remotepath, filename)

    local_path = set_path_classif(path_to_classif)

    with paramiko.SSHClient() as sshclient:
        sshclient.load_system_host_keys()
        _connect_ssh(sshclient, hostname)

        extensions = ["classif"]
        if download_statedicts:
            extensions += ["tar", "grads"]

        progress = _ProgressBar()
        with scp.SCPClient(
            sshclient.get_transport(), progress=progress
        ) as scpclient:
            for ext in extensions:
                filenameext = '.'.join([filename, ext])
                try:
                    scpclient.get(
                        filenameext, local_path=local_path, preserve_times=True
                    )
                except scp.SCPException:
                    pass
                progress.newline()


def _get_dbentry(
        index_db, path_to_csv=None, path_to_classif=None
):
    # Load object reference database
    path_to_classif = set_path_classif(path_to_classif)
    try:
        df, _ = load_db(path_to_csv)
    except FileNotFoundError as exc:
        raise CSVNotFoundError("CSV file not found.") from exc
    dbentry = df.loc[index_db]
    if not isinstance(dbentry, pd.core.series.Series):
        dbentry = dbentry.squeeze(axis=0) # Convert into Pandas series

    return dbentry


def _connect_ssh(sshclient, hostname, configfile="~/.ssh/config"):

    # Adapted from https://www.programcreek.com/python/example/52881/paramiko.ProxyCommand
    # Source Project: pyfg
    # Author: spotify
    # File: fortios.py
    # License: Apache License 2.0
    sshclient.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if os.path.exists(os.path.expanduser(configfile)):
        ssh_config = paramiko.SSHConfig()
        user_config_file = os.path.expanduser(configfile)
        with io.open(user_config_file, 'rt', encoding='utf-8') as f:
            ssh_config.parse(f)

        host_conf = ssh_config.lookup(hostname)
        kwargs = {}
        if host_conf:
            if 'hostname' in host_conf:
                hostname = host_conf['hostname']
            if 'proxycommand' in host_conf:
                kwargs.update(sock=paramiko.ProxyCommand(host_conf['proxycommand']))
            if 'user' in host_conf:
                kwargs.update(username=host_conf['user'])
            if 'identityfile' in host_conf:
                kwargs.update(key_filename=host_conf['identityfile'])

    sshclient.connect(hostname, **kwargs)


def _check_consistency_and_repair(classif, exclude=None, warn=True):
    """
    Check that the loaded classifier has all the attributes given as
    constructor parameters (for backward compatibility).

    Parameters
    ----------
    exclude (list of str): attributes to exclude from checking

    """
    # Constructor parameters
    params = inspect.signature(classif.__class__).parameters
    params = dict(params)

    # Delete unwanted parameters
    if exclude is not None:
        for par in exclude:
            del params[par]
    
    # Replace attribute name (backward compatibility)
    for old_attrname, new_attrname in OLDATTRS.items():
        if old_attrname in classif.__dict__:
            val = getattr(classif, old_attrname)
            setattr(classif, new_attrname, val)
            delattr(classif, old_attrname)
            if warn:
                warnings.warn(
                    "Attribute name '{}' changed: new name = '{}', set to {}.".format(
                        old_attrname, new_attrname, val
                    )
                )

    # Check attributes
    for attr_name in params:

        # Set missing attributes to their default value
        if attr_name not in classif.__dict__:
            attr_val = params[attr_name].default
            if attr_val is not inspect.Parameter.empty:
                setattr(classif, attr_name, attr_val)
                # Warning message
                if warn:
                    msg = "Attribute '{}' not found; set to {}.".format(
                        attr_name, attr_val
                    )
                    warnings.warn(msg)
            else:
                raise AttributeError(
                    "Missing default value for parameter '{}'.".format(attr_name)
                )
        # Replace attribute values, if needed
        elif attr_name in ATTR_SUBSTITUTION.keys():
            old_attrval = getattr(classif, attr_name)
            if old_attrval in ATTR_SUBSTITUTION[attr_name].keys():
                new_attrval = ATTR_SUBSTITUTION[attr_name][old_attrval]
                setattr(classif, attr_name, new_attrval)
                if warn:
                    warnings.warn(
                        "Value for attribute '{}' changed: old value = {}, "
                        "new value = {}.".format(
                            attr_name, old_attrval, new_attrval
                        )
                    )

    # Replace attribute types
    if 'lr_scheduler_class' in classif.__dict__:
        _update_attr_backward_compatibility(
            classif, LR_SCHEDULER_DICT, 'lr_scheduler',
            old_attr='lr_scheduler_class'
        )
    if not (classif.metrics is None or isinstance(classif.metrics, str)):
        _update_attr_backward_compatibility(
            classif, metr.METRICS_DICT, 'metrics'
        )


def set_path(path, arg_name):
    if path is None:
        if CONFIGDICT is not None:
            path = CONFIGDICT["classifiers"][arg_name]
        if path is None:
            raise ValueError("Missing value for argument '{}'.".format(arg_name))
    return os.path.expanduser(path)

def set_path_classif(path):
    return set_path(path, "path_to_classif")

def set_path_csv(path):
    return set_path(path, "path_to_csv")


_DB_FILENAME = 'classif_db.csv'
_LEN_INDEX = 2

def load_db(path_to_csv=None):
    if path_to_csv is None:
        path_to_csv = set_path_csv(path_to_csv)
    db_filename = _DB_FILENAME
    len_index = _LEN_INDEX
    df = pd.read_csv(
        os.path.join(path_to_csv, db_filename), index_col=list(range(len_index)),
        keep_default_na=False
    )
    return df, db_filename

def save_db(df, path_to_csv=None):
    if path_to_csv is None:
        path_to_csv = set_path_csv(path_to_csv)
    db_filename = _DB_FILENAME
    df.to_csv(os.path.join(path_to_csv, db_filename))


def get_filename_and_update_db(
        index_db, path_to_csv=None, **kwargs
):
    # Load CSV database
    df, _ = load_db(path_to_csv)

    # Select entry in the database or create a new one
    try:
        filename = df.loc[index_db].filename

    except KeyError:
        filename = uuid.uuid4().hex # New file name (unique identifier)
        dbentry = pd.Series(
            data=dict(filename=filename, **kwargs), # Fields
            name=index_db # Index
        )
        # New entry in the classifier reference database
        df = df.append(dbentry, verify_integrity=True)
        save_db(df, path_to_csv)

    return filename


def get_default_metrics(classif, eval_mode):

    if eval_mode in ('shifts', 'perturbations'):
        metrics = 'mfr' # Change default value
    else:
        metrics = classif.metrics

    return metrics


class Score:

    def __init__(self, val, commit=None):
        self.val = val
        self.commit = commit
        self.date = datetime.now()

    def __call__(self):
        return self.val

    def __repr__(self):

        out = self()
        if isinstance(out, torch.Tensor):
            out = "tensor of shape {}".format(tuple(out.shape))

        out = "Value = {}\n".format(out)
        out += "Registration date = {}\n".format(self.date.strftime("%Y-%m-%d %H:%M:%S"))
        if self.commit is not None:
            out += "Commit = {}".format(self.commit)

        return out

class ClassifNotFoundError(FileNotFoundError):
    pass

class CSVNotFoundError(FileNotFoundError):
    pass
