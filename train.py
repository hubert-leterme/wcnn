import argparse
import warnings
import inspect
import ipdb

from wcnn.classifier import WaveCnnClassifier
from wcnn.classifier.classifier_toolbox import ClassifNotFoundError
from wcnn import utils, toolbox, data
from wcnn.cnn import models
from wcnn.utils import LIST_OF_KWARGS_DATASET, LIST_OF_KWARGS_DIST

from base_scripts import base

# Dictionary of dataset-dependent arguments to be passed to the constructor of
# the classifier (only the arguments which differ from the default arguments).
DATASET_DEPENDENT_ARGS = {
    'MNIST': dict(size=32, data_augmentation=False),
    'ImageNet': dict(metrics='top1_5_accuracy_scores'),
    'CIFAR10': dict(size=32)
}

# List of constructor parameters
LIST_OF_KWARGS_CLASSIF = inspect.getfullargspec(
    WaveCnnClassifier
).args[1:] # Remove "self"
LIST_OF_KWARGS_CLASSIF.remove("name")
LIST_OF_KWARGS_SAVE = ['save_state_dicts', 'save_best_classif']

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'name',
        help="Name used as an identifier."
    )
    parser.add_argument(
        '-a', '--arch',
        default=argparse.SUPPRESS,
        help=("Model architecture: {}.".format(" | ".join(models.ARCH_LIST)))
    )
    parser.add_argument(
        '-c', '--config',
        default=argparse.SUPPRESS,
        help=("Describes the network's configuration. See documentation related "
              "to the chosen CNN architecture for more details. Optional if "
              "training is resumed from an existing classifier. Otherwise, if "
              "none is given, then the default value will be used.")
    )
    parser.add_argument(
        '-r', '--resume-from-snapshot',
        type=int,
        default=None,
        help=("Specify the training epoch from which resuming training. If none is "
              "given, training will be resumed from the latest checkpoint (or from scratch"
              "if nonexisting).")
    )
    base.parser_addarguments_dataset(parser, default_split='train')
    parser.add_argument(
        '--turn-off-data-augm',
        action='store_true',
        help="Turn off data augmentation."
    )
    parser.add_argument(
        '--get-gradients',
        action='store_true',
        help=("During training, the l-infty norm of some gradients will be "
              "recorded for each batch. The weight tensors with respect to which "
              "the gradients are recorded are provided by attribute "
              "'weights_l1reg' of the "
              "chosen model. The purpose of this is to estimate a suitable "
              "value of the hyperparameter lambda when performing L1 "
              "regularization.")
    )
    parser.add_argument(
        '--has-l1loss',
        action='store_true',
        help=("Regularize the loss function with L1 norm on the weights used "
              "for filter selection (the weights are provided by the chosen "
              "model).")
    )
    parser.add_argument(
        '--lambda-params',
        type=float,
        nargs='*',
        default=argparse.SUPPRESS,
        help="Hyperparameter for the regularizer."
    )
    parser.add_argument(
        '-lr', '--lr',
        type=float,
        default=argparse.SUPPRESS,
        help="Initial learning rate. Default = 0.1"
    )
    parser.add_argument(
        '--lr-scheduler',
        default=argparse.SUPPRESS,
        help=("Learning rate scheduler. Default = 'StepLR30'")
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=argparse.SUPPRESS,
        help="Momentum factor for SGD. Default = 0.9"
    )
    parser.add_argument(
        '-wd', '--weight-decay',
        type=float,
        default=argparse.SUPPRESS,
        help="Weight decay (L2 penalty) for SGD. Default = 5e-4"
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=argparse.SUPPRESS,
        help="Number of total epochs to run. Default = 90"
    )
    parser.add_argument(
        '-b', '--batch-size-train',
        type=int,
        default=argparse.SUPPRESS,
        help=("Mini-batch size for training, this is the total batch size of all GPUs "
              "on the current node when using Data Parallel or Distributed "
              "Data Parallel. Default = 256")
    )
    parser.add_argument(
        '--batch-size-test',
        type=int,
        default=argparse.SUPPRESS,
        help=("Mini-batch size for evaluation, this is the total batch size of all GPUs "
              "on the current node when using Data Parallel or Distributed "
              "Data Parallel. Default = 256")
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=None,
        help=("Number of images in the truncated training set. Default = None "
              "(whole dataset)")
    )
    parser.add_argument(
        '--n-val',
        type=int,
        default=None,
        help=("Number of images in the truncated validation set. Default = None "
              "(whole dataset)")
    )
    parser.add_argument(
        '-ba', '--batch-accum',
        type=int,
        default=argparse.SUPPRESS,
        help=("Performs gradient accumulation with this number of batches. Arguments "
              "'batch_size_train' and 'batch_size_test' should not be changed; they "
              "are automatically adjusted. Default = 1. More details on "
              "https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-"
              "to-zero-in-pytorch/4903/20?u=alband")
    )
    parser.add_argument(
        '--accumulate-grads-insteadof-averaging',
        action='store_true',
        default=argparse.SUPPRESS,
        help=""
    )
    parser.add_argument(
        '--turn-off-validation',
        action='store_true',
        help="Do not use any validation set."
    )
    parser.add_argument(
        '--wavelet',
        default=argparse.SUPPRESS,
        help="Name of the wavelet, if required. Ex. 'haar', 'db3', 'qshift_a'..."
    )
    parser.add_argument(
        '--backend',
        default=argparse.SUPPRESS,
        help=("One of 'pywt' or 'dtcwt'. Name of the library from which to get "
              "the filters. Default = 'dtcwt'")
    )
    parser.add_argument(
        '--trainable-qmf',
        action='store_true',
        default=argparse.SUPPRESS,
        help=("Set the QMFs as trainable parameters.")
    )
    parser.add_argument(
        '--pretrained-model',
        default=argparse.SUPPRESS,
        help=("Name of the classifier containing the pretrained model. If None "
              "is given, then the training is made from scratch.")
    )
    parser.add_argument(
        '--status-pretrained',
        default=argparse.SUPPRESS,
        help=("Status of the pretrained classifier. Can be one of 'checkpoint', "
              "'best' or 'e<epochs>', where <epochs> denotes the number of "
              "loops over the whole training set, during the training phase.")
    )
    parser.add_argument(
        '--modules-initialized-from-scratch',
        default=argparse.SUPPRESS,
        help=("If '--pretrained_model' is provided, then this argument indicates "
              "whether some weights shall nevertheless be initialized from "
              "scratch. Possible values depend on the chosen model. This "
              "parameter was first introduced to distinguish low- from high-pass "
              "filters in DtCwptAlexNet.")
    )
    parser.add_argument(
        '--padding-mode',
        default=argparse.SUPPRESS,
        help=("One of 'zeros', 'symmetric' or 'reflect'. If none is given, then "
              "the default value depends on the specified network architecture.")
    )
    parser.add_argument(
        '--snapshots',
        type=int,
        default=argparse.SUPPRESS,
        help=("If not null, a partially trained classifier will be saved every "
              "p epochs, p being given by this parameter. Default = 25")
    )
    parser.add_argument(
        '--save-best-classif',
        action='store_true',
        default=argparse.SUPPRESS,
        help=("Save a snapshot each time the classifier reaches its best accuracy "
              "so far.")
    )
    parser.add_argument(
        '--ignore-attr-consistency',
        action='store_true',
        help=("Do not check commit number and attribute consistency when loading an "
              "existing classifier. TO BE USED WITH CAUTION to avoid reproducibility "
              "issues.")
    )
    base.parser_addarguments_distributed(parser)
    base.parser_addarguments_infodebug(parser)

    args = parser.parse_args()

    # Set debug mode
    if args.debug % 2 == 0:
        args.ignore_attr_consistency = True
        # In debug mode, a fake dataset is used with many less images. Therefore,
        # the number of images per class in the validation set is set to a much
        # smaller number than for the actual dataset.
        if args.ds_name is None or "ImageNet" in args.ds_name:
            args.n_imgs_per_class_val = 8
        args.n_train = 16
        args.n_val = 16
        args.batch_size_train = 4
        args.batch_size_test = 4
        args.verbose_print_batch = 1
    if args.debug % 3 == 0:
        ipdb.set_trace()

    if args.turn_off_data_augm:
        args.data_augmentation = False

    utils.spawn_mainworker(main_worker, args)


def main_worker(gpu, ngpus_per_node, args):

    infoprinter = utils.set_distributed_process(gpu, ngpus_per_node, args)

    kwargs = vars(args).copy()

    # Dataset-dependent arguments
    kwargs_dataset = {}
    try:
        kwargs_dataset = DATASET_DEPENDENT_ARGS[args.ds_name]
    except KeyError:
        for key in DATASET_DEPENDENT_ARGS:
            if key in args.ds_name:
                kwargs_dataset = DATASET_DEPENDENT_ARGS[key]
                break
    kwargs.update(**kwargs_dataset)

    # Load dataset
    dataloader = data.__dict__[args.ds_name]
    kwargs_dataset = toolbox.extract_dict(LIST_OF_KWARGS_DATASET, kwargs)

    ## Training set
    infoprinter("Loading training dataset...")
    ds = utils.load_dataset(
        dataloader, n_imgs=args.n_train, infoprinter=infoprinter,
        **kwargs_dataset
    )
    infoprinter("...done.")
    ## Validation set
    if not args.turn_off_validation:
        infoprinter("Loading validation set...")
        kwargs_dataset.update(split='val')
        ds_val = utils.load_dataset(
            dataloader, n_imgs=args.n_val, infoprinter=infoprinter,
            task="validation", **kwargs_dataset
        )
        infoprinter("...done.")
    else:
        ds_val = None
        warnings.warn("No validation will be performed.")

    # Keyword arguments
    kwargs_classif = toolbox.extract_dict(LIST_OF_KWARGS_CLASSIF, kwargs)
    kwargs_fit = {}
    kwargs_dist = toolbox.extract_dict(LIST_OF_KWARGS_DIST, kwargs)
    kwargs_load = {}
    kwargs_save = toolbox.extract_dict(LIST_OF_KWARGS_SAVE, kwargs)

    if args.resume_from_snapshot is not None:
        kwargs_load.update(
            status='e{}'.format(args.resume_from_snapshot)
        )
    if args.get_gradients: # 'get_gradients' is also a constructor parameter
        kwargs_load.update(load_grads=True)
    if args.commit is not None:
        kwargs_load.update(commit=args.commit)
        kwargs_save.update(commit=args.commit)

    # Load an existing classifier and resume training...
    try:
        infoprinter("Attempting loading classifier...")
        if args.reset_workers:
            kwargs_load.update(workers=None)
        classif = WaveCnnClassifier.load(
            id_classif=args.name,
            ignore_attr_consistency=args.ignore_attr_consistency,
            **kwargs_load, **kwargs_classif
        )
        kwargs_fit.update(resume=True)
        infoprinter("...done.")

    # ... or create new classifier
    except KeyError:
        infoprinter("...not found. Initializing classifier...")
        classif = WaveCnnClassifier(name=args.name, **kwargs_classif)
        infoprinter("...done.")

    except ClassifNotFoundError as err:
        raise ClassifNotFoundError(
            "A checkpoint was previously created but the "
            "corresponding file is missing. You can use another name "
            "for the classifier, or remove the existing entry "
            "in classif_db.csv."
        ) from err

    if args.debug > 1:
        name_debug = '{}_debug'.format(args.name)
        classif.name = name_debug

    script, args_script = utils.get_script_and_args()
    kwargs_save.update(script=script, args_script=args_script)
    infoprinter("Starting training...")
    classif.fit(
        ds, ds_val, **kwargs_fit, **kwargs_dist, **kwargs_save
    )


if __name__ == "__main__":
    main()
