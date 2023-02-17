from sklearn.base import BaseEstimator

from ._mixin_classes import TopNClassifierMixin, TrainerMixin, PredictorMixin, \
                            SaverLoaderMixin

from .. import data

class WaveCnnClassifier(SaverLoaderMixin, PredictorMixin, TrainerMixin,
                        TopNClassifierMixin, BaseEstimator):
    """
    A classifier based on wavelet CNNs, using PyTorch backend. Leaning is done
    by using stochastic gradient descent with specified learning rate, momentum
    and weight decay.

    Parameters
    ----------
    name (str, default=None)
        Used as an identifier.
    size (int, default=224)
        Size of the input matrices. Before training, images are resized to
        this value.
    normalize (bool, default=True)
        Whether to normalize the data before training. See
        https://pytorch.org/docs/master/torchvision/models.html for more details.
    data_augmentation (bool, default=True)
        If True, then input images are randomly flipped and cropped from 8% to
        100% of their original sizes, with a random aspect ratio varying from
        3/4 to 4/3, before being resized using a bilinear interpolation.
    metrics (str, default='top1_accuracy_score')
        Metrics used to compute the score. It is used as a key in
        metrics.METRICS_DICT.
    arch (str, default=None)
        Name of the CNN architecture. It is used as a key in dictionary
        wcnn.cnn.models.__dict__, from which the network class is retrieved.
    config (str, default=None)
        String describing the network's configuration. See documentation related
        to the chosen CNN architecture for more details.
    wavelet (str, default=None)
        Initial wavelet.
    backend (str, default=None)
        Name of the library from which to get the wavelet filters. Can be one of
        'pywt' or 'dtcwt'. The default backend depends on the specified network
        architecture.
    ds_name (str, default='ImageNet')
    split (str, default='train')
    n_train, n_val (int, default=None)
    pretrained_model (str, default=None)
        Path to the classifier containing the pretrained model. If None is given,
        then the training is made from scratch.
    status_pretrained (str, default=None)
        Status of the pretrained model. Can be one of 'checkpoint', 'best' or
        'e<epochs>', where <epochs> denotes the number of loops over the
        whole training set, during the training phase.
    padding_mode (str, default=None)
        One of 'zeros', 'symmetric' or 'reflect'. If none is given, then the
        default value depends on the specified network architecture.
    get_gradients (bool, default=False)
        If set to True, then, during training, the l-infty norm of some gradients
        will be recorded for each batch. The weight tensors with respect to which
        the gradients are recorded are provided by attribute 'weights_l1reg' of
        the chosen model.
        The purpose of this is to estimate a suitable value of the hyperparameter
        lambda when performing L1 regularization.
    has_l1loss (bool, default=False)
        Whether to regularize the loss function with L1 norm on the weights used
        for filter selection (the weights are provided by the chosen model,
        stored as attribute 'weights_l1reg').
    lambda_params (list of float, default=None)
        List of hyperparameters for the regularizer. Its length must match the
        length of 'net.weights_l1reg', where 'net' denotes the chosen model.
    lr (float, default=0.01)
        Initial learning rate.
    lr_scheduler (str, default='StepLR30')
        Name of the learning rate scheduler. It is used as a key in
        classifier.LR_SCHEDULER_DICT, from which the scheduler class and initial
        settings are retrieved. See torch.optim for more details. If none is
        given, then no learning rate scheduling will be applied.
    momentum (float, default=0.9)
    weight_decay (float, default=5e-4)
    epochs (int, default=100)
        Number of epochs during the training phase.
    batch_size_train (int, default=256)
        Batch size during the training phase.
    suffle_data (bool, default=True)
        Whether to sample images randomly from the training set.
    workers (int, default=None)
        How many subprocesses to use for data loading. 0 means that the data
        will be loaded in the main process. More details on
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        If none is given, the number of workers will be inferred from the
        number of available CPUs.
    batch_size_test (int, default=256)
        Batch size during the validation phase.
    batch_accum (int, default=1)
        Performs gradient accumulation with this number of batches. Arguments
        'batch_size_train' and 'batch_size_test' should not be changed; they
        are automatically adjusted. More details on
        https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20?u=alband
    verbose_print_batch (int, default=50)
        Print info every xx batch.
    verbose (bool, default=False)
        Print additional information, for debug.
    snapshots (int, default=10)
        If > 0, a partially trained classifier will be saved every p epochs, p
        being given by this parameter.

    """
    def __init__(
            self, name=None, size=224, normalize=True, data_augmentation=True,
            metrics='top1_accuracy_score', arch=None, config=None, wavelet=None,
            backend=None, ds_name='ImageNet', split='train', n_train=None,
            n_val=None, pretrained_model=None, status_pretrained=None,
            padding_mode=None, get_gradients=False, has_l1loss=False, lambda_params=None,
            lr=0.1, lr_scheduler='StepLR30', momentum=0.9, weight_decay=1e-4,
            epochs=90, batch_size_train=256, shuffle_data=True,
            workers=None, batch_size_test=256, batch_accum=1,
            accumulate_grads_insteadof_averaging=False, verbose_print_batch=50,
            verbose=False, snapshots=30
    ):
        self.name = name
        self.size = size
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.metrics = metrics
        self.arch = arch
        self.config = config
        self.wavelet = wavelet
        self.backend = backend
        self.ds_name = ds_name
        self.split = split
        self.n_train = n_train
        self.n_val = n_val
        self.pretrained_model = pretrained_model
        self.status_pretrained = status_pretrained
        self.padding_mode = padding_mode
        self.get_gradients = get_gradients
        self.has_l1loss = has_l1loss
        self.lambda_params = lambda_params
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size_train = batch_size_train
        self.shuffle_data = shuffle_data
        self.workers = workers
        self.batch_size_test = batch_size_test
        self.batch_accum = batch_accum
        self.accumulate_grads_insteadof_averaging = accumulate_grads_insteadof_averaging
        self.verbose_print_batch = verbose_print_batch
        self.verbose = verbose
        self.snapshots = snapshots


    def __eq__(self, other):
        # TODO Remove (deprecated)
        try:
            out = self.name == other.name and \
                  self.current_epoch_ == other.current_epoch_ and \
                  self.date_and_time_per_epoch_ == other.date_and_time_per_epoch_
        except AttributeError:
            # Classifier not initialized (e.g., network fully deterministic)
            out = self.name == other.name

        return out


    def _check_dataset_integrity(self, dataset):
        if type(dataset) != data.__dict__[self.ds_name]:
            raise ValueError("Wrong dataset type: `{}` expected; got `{}`.".format(
                data.__dict__[self.ds_name], type(dataset)
            ))
