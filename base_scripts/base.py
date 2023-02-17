import argparse

from wcnn import data

LIST_OF_KWARGS_METHOD = [
    'max_shift', 'pixel_divide', 'path_to_pred', 'return_ref', 'commit',
    'script', 'args_script'
]

DS_NAMES = sorted(
    name for name, obj in data.__dict__.items()
    if not name.startswith("__")
    and callable(obj)
)

def parser_addarguments_dataset(parser, default_split):

    parser.add_argument(
        '-ds', '--ds-name',
        default='ImageNet',
        help="Dataset: {}. Default = ImageNet (ILSVRC2012)".format(
            " | ".join(DS_NAMES)
        )
    )
    parser.add_argument(
        '--root',
        default=argparse.SUPPRESS,
        help=("Path to the dataset. If none is given, a config file "
              "'wcnn_config.yml' must be provided.")
    )
    parser.add_argument(
        '--split',
        default=default_split,
        help=("'train', 'val' or 'test'. Default = '{}'.".format(default_split))
    )
    parser.add_argument(
        '--size',
        type=int,
        default=argparse.SUPPRESS,
        help=("Size of the input images. If none is given, the default size "
              "depends on the dataset (224 for ImageNet).")
    )
    parser.add_argument(
        '-j', '--workers',
        type=int,
        default=argparse.SUPPRESS,
        help=("Number of data loading workers. If none is given, it is "
              "inferred from the number of available CPUs.")
    )
    parser.add_argument(
        '--reset-workers',
        action='store_true',
        help="Sets parameter 'workers' to None in the classifier."
    )


def parser_addarguments_distributed(parser):

    parser.add_argument(
        '--world-size',
        default=1,
        type=int,
        help='Number of nodes for distributed training. Default = 1'
    )
    parser.add_argument(
        '--rank',
        default=0,
        type=int,
        help='Node rank for distributed training. Default = 0'
    )
    parser.add_argument(
        '--dist-url',
        default=None,
        type=str,
        help="URL used to set up distributed training. Default = 'env://'"
    )
    parser.add_argument(
        '--dist-backend',
        default='nccl',
        type=str,
        help="Distributed backend. Default = 'nccl'"
    )


def parser_addarguments_infodebug(parser):

    parser.add_argument(
        '--verbose-print-batch',
        type=int,
        default=argparse.SUPPRESS,
        help="Print info every xx batch. Default = 50"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Print additional information, for debug."
    )
    parser.add_argument(
        '--commit',
        default=None,
        help="Hash of the current git commit. Used for the sake of replicability."
    )
    parser.add_argument(
        '-d', '--debug',
        type=int,
        default=1,
        help=("1: normal mode (default value);\n"
              "2: use very small dataset and very small batches for debugging "
              "purpose;\n"
              "3: use PDB debugger;\n"
              "6: combination of 2 and 3.")
    )
