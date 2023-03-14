import argparse
from wcnn import utils

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'idx',
        nargs='*',
        help="Database entry to remove."
    )
    parser.add_argument(
        '--path-to-csv',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--path-to-classif',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--remove-files',
        action='store_true',
        default=argparse.SUPPRESS,
        help="Remove corresponding files."
    )

    args = parser.parse_args()
    kwargs = vars(args).copy()
    kwargs.pop('idx')
    utils.remove_classif(args.idx, **kwargs)


if __name__ == "__main__":
    main()
