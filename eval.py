"""
Compute the accuracy score for a trained classifier, on a given dataset.

"""
import argparse
import ipdb

from wcnn import utils, data, toolbox
from wcnn.classifier import WaveCnnClassifier
from wcnn.utils import LIST_OF_KWARGS_DATASET, LIST_OF_KWARGS_DIST
from wcnn.classifier.classifier_toolbox import LIST_OF_KWARGS_FORWARDEVAL

from base_scripts import base

LIST_OF_KWARGS_LOAD = [
    'status', 'use_old_confignames', 'backward_compatibility', 'modulus'
]
LIST_OF_KWARGS_SAVE = ['commit', 'overwrite', 'path_to_classif', 'ds_name', 'split']

LIST_OF_KWARGS_SCORE = [
    'eval_mode', 'metrics', 'max_step_size'
]

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'classif',
        help="Classifier to be evaluated."
    )
    parser.add_argument(
        '-s', '--status',
        default=argparse.SUPPRESS,
        help=("Status of the classifier. One of 'checkpoint', 'best' or "
              "'e<epochs>'. Default = 'checkpoint'")
    )
    parser.add_argument(
        '--use-old-confignames',
        action='store_true',
        default=argparse.SUPPRESS,
        help="Use old config names, for backward compatibility."
    )
    base.parser_addarguments_dataset(parser, default_split='test')
    parser.add_argument(
        '--n-imgs',
        type=int,
        default=argparse.SUPPRESS,
        help="Default = None (all test images)"
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=argparse.SUPPRESS,
        help=("Mini-batch size for evaluation, this is the total batch size of all GPUs "
              "on the current node when using Data Parallel or Distributed "
              "Data Parallel. Default = 256")
    )
    parser.add_argument(
        '-em', '--eval-mode',
        default='onecrop',
        help="'onecrop', 'tencrops', 'shifts', or 'perturbations'. Default = 'onecrop'"
    )
    parser.add_argument(
        '--max-shift',
        type=int,
        default=argparse.SUPPRESS,
        help=("Maximum number of pixels in each direction, for image shift. "
              "Must be provided if '--eval-mode' is set to 'shifts'.")
    )
    parser.add_argument(
        '--pixel-divide',
        type=int,
        default=argparse.SUPPRESS,
        help=("Number of intermediate shifts per pixel. For example, if set to "
              "2, then images will be shifted by steps of 0.5 pixels. This is "
              "made possible by using bilinear interpolation and antialiased "
              "downsampling.")
    )
    parser.add_argument(
        '--shift-direction',
        default=argparse.SUPPRESS,
        help=("One of 'vertical', 'horizontal' or 'diagonal'. If none is given, then shifts "
              "will be computed along both directions, which may lead to high memory consumption.")
    )
    parser.add_argument(
        '-m', '--metrics',
        default=argparse.SUPPRESS,
        help=("One of 'kldiv' (KL divergence between outputs) or 'mfr' (mean flip rate "
              "between any two frames with distance ranging from 1 to --max-step-size. "
              "If none is given, then the metrics is chosen as follows:\n"
              "\t- eval_mode == 'onecrop' or 'tencrops' >> provided by attribute self.metrics_;\n"
              "\t- eval_mode == 'shifts' or 'perturbations' >> mean flip rate (equivalent to "
              "metrics='mfr').")
    )
    parser.add_argument(
        '--max-step-size',
        type=int,
        default=argparse.SUPPRESS,
        help="Step size when the mean flip rate is computed (see --metrics). Default = 5"
    )
    parser.add_argument(
        '--backward-compatibility',
        action='store_true',
        default=argparse.SUPPRESS,
        help=("Applies patch following commits nb 1ef262b322fab5ee4e0a31a8d181aba4748c34d4 "
              "and 4bd8683ec343b89381a0bd4f5b9299a6687eb946 when loading state dicts.")
    )
    parser.add_argument(
        '--modulus',
        action='store_true',
        default=argparse.SUPPRESS,
        help=("Whether the network implements CMod instead of RMax. Only "
              "valid if argument '--backward-compatibility' is specified.")
    )
    base.parser_addarguments_distributed(parser)
    parser.add_argument(
        '-f', '--overwrite',
        action='store_true',
        default=argparse.SUPPRESS,
        help=("Overwrites existing recorded score.")
    )
    base.parser_addarguments_infodebug(parser)

    args = parser.parse_args()

    # Set debug mode
    if args.debug % 2 == 0:
        if args.ds_name is None or "ImageNet" in args.ds_name:
            args.n_imgs_per_class_val = 8
        args.n_imgs = 16
        args.batch_size = 2
        args.verbose_print_batch = 1
        args.path_to_classif = "./" # Save score in a separate file
    if args.debug % 3 == 0:
        ipdb.set_trace()

    utils.spawn_mainworker(main_worker, args)


def main_worker(gpu, ngpus_per_node, args):

    infoprinter = utils.set_distributed_process(gpu, ngpus_per_node, args)

    if args.eval_mode == 'onecrop':
        infoprinter(
            "Evaluation done on a center patch of the resized image.", always=True
        )
    elif args.eval_mode == 'tencrops':
        infoprinter(
            "Evaluation done by averaging outputs over 10 patches.", always=True
        )
    elif args.eval_mode == 'shifts':
        infoprinter(
            "Evaluation done on several shifted versions of a single image.",
            always=True
        )

    kwargs = vars(args).copy()

    # Load dataset
    dataloader = data.__dict__[args.ds_name]
    kwargs_dataset = toolbox.extract_dict(LIST_OF_KWARGS_DATASET, kwargs, pop=False)
    # We have set pop=False because some arguments are used several times, e.g., 'split'.

    infoprinter("Loading evaluation set...")
    dataset = utils.load_dataset(
        dataloader, infoprinter=infoprinter, task="evaluation",
        **kwargs_dataset
    )
    infoprinter("...done.")

    K = len(dataset)

    # Other keyword arguments
    kwargs_score = toolbox.extract_dict(LIST_OF_KWARGS_SCORE, kwargs, pop=False)
    kwargs_forward = toolbox.extract_dict(LIST_OF_KWARGS_FORWARDEVAL, kwargs, pop=False)
    kwargs_dist = toolbox.extract_dict(LIST_OF_KWARGS_DIST, kwargs, pop=False)
    kwargs_load = toolbox.extract_dict(LIST_OF_KWARGS_LOAD, kwargs, pop=False)
    kwargs_save = toolbox.extract_dict(LIST_OF_KWARGS_SAVE, kwargs, pop=False)

    # Load classifier
    if args.reset_workers:
        kwargs_load.update(workers=None)
    classif = WaveCnnClassifier.load(
        args.classif, train=False, **kwargs_load
    )

    # Print info
    device = classif.device_train_
    try:
        traindate = classif.training_date_and_time_.strftime("%Y-%m-%d")
        traintime = classif.training_date_and_time_.strftime("%H:%M:%S")
    except AttributeError:
        traindate = '?'
        traintime = '?'

    infoprinter(
        "Classifier: {}, trained on {}, {} at {}".format(
            classif.name, device, traindate, traintime
        ), always=True
    )
    infoprinter(
        "Number of trained epochs =", classif.current_epoch_, always=True
    )
    infoprinter(
        "The evaluation will be made on the {} set, with {} images.".format(
            args.split, K
        ), always=True
    )
    infoprinter("", always=True)
    infoprinter("Model = {}".format(classif.arch), always=True)
    infoprinter("Config = {}".format(classif.config), always=True)
    infoprinter(
        "Batch size for the training = {}".format(classif.batch_size_train),
        always=True
    )
    infoprinter(
        "Training stopped after {} epochs".format(classif.current_epoch_),
        always=True
    )
    infoprinter("", always=True)

    # Perform evaluation
    classif.training_in_progress_ = False
    if args.debug > 1:
        classif.name = "{}_debug".format(classif.name)
    infoprinter("Starting evaluation...")
    classif.score(
        dataset, save_results=True, **kwargs_score, **kwargs_forward,
        **kwargs_dist, **kwargs_save
    )
    infoprinter("...done.")


if __name__ == "__main__":
    main()
