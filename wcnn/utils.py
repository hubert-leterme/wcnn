"""
The functions defined here are only intended for external usage. Any function
that may be imported by other modules must be placed in the toolbox module.
This is done to prevent circular imports.

"""
import os, sys, traceback, socket
import ipdb
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd

from . import toolbox
from . import classifier as cl

from . import N_GPUS

#=================================================================================
# Functions used in the scripts provided in the repository
#=================================================================================

LIST_OF_KWARGS_DATASET = [
    'root', 'split', 'n_imgs_per_class_train', 'n_imgs_per_class_val', 'annfile',
    'trainfolder', 'grayscale', 'n_imgs'
]
LIST_OF_KWARGS_DIST = ['distributed', 'gpu', 'rank', 'world_size']


def get_script_and_args():
    script = sys.argv[0].split('/')[-1]
    args_script = ' '.join(sys.argv[1:])
    return script, args_script


def remove_classif(
        index_db, path_to_csv=None, path_to_classif=None, remove_files=False
):
    # Load object reference database and get list of filenames
    path_to_csv = cl.classifier_toolbox.set_path_csv(path_to_csv)
    df, _ = cl.classifier_toolbox.load_db(path_to_csv)
    filenames = df.loc[index_db].filename
    if isinstance(filenames, pd.Series):
        filenames = filenames.tolist()
    elif isinstance(filenames, str):
        filenames = [filenames]

    # Update database
    df = df.drop(index=index_db)
    cl.classifier_toolbox.save_db(df, path_to_csv)

    # Remove objects
    if remove_files:
        path_to_classif = cl.classifier_toolbox.set_path_classif(path_to_classif)
        listdir = [
            fn for fn in os.listdir(path_to_classif) \
                    if os.path.isfile(os.path.join(path_to_classif, fn))
        ]
        for fn in listdir:
            if fn.split('.')[0] in filenames:
                os.remove(os.path.join(path_to_classif, fn))


def load_dataset(
        dataloader, n_imgs=None, infoprinter=print,
        task="training", **kwargs
):
    # Training set
    dataset = dataloader(**kwargs)
    if n_imgs is not None:
        dataset.truncate(n_imgs, update_classes=False)
        dataset.debug = True # Debug mode ("fake prediction")
    infoprinter("Dataset used for {}:".format(task), always=True)
    infoprinter(dataset, always=True)

    return dataset


def spawn_mainworker(main_worker, args):

    if N_GPUS is not None:
        ngpus_per_node = N_GPUS
    else:
        ngpus_per_node = 1
    args.multiprocessing_distributed = (ngpus_per_node > 1)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.distributed and args.dist_url is None:
        freeport = _get_freeport()
        args.dist_url = "tcp://localhost:" + str(freeport)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)
        )
    else:
        # Simply call main_worker function (gpu=None)
        main_worker(None, ngpus_per_node, args)


def _get_freeport():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr = s.getsockname()
    s.close()
    return addr[1]


def set_distributed_process(gpu, ngpus_per_node, args):

    args.gpu = gpu

    infoprinter = toolbox.InfoPrinter(
        distributed=args.distributed, gpu=args.gpu,
        verbose=args.verbose
    )

    if args.gpu is not None:
        infoprinter("Hello.")

    if args.distributed:
        infoprinter("Communicating with", args.dist_url)
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        infoprinter("Initializing process group...")
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank
        )
        infoprinter("...done.")

    return infoprinter


#=================================================================================
# Miscellaneous functions
#=================================================================================

def debug_on_error(fun, *args, **kwargs):
    try:
        fun(*args, **kwargs)
    except:
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)


def find_duplicates_db(path_to_csv=None):
    df, _ = cl.classifier_toolbox.load_db(path_to_csv)
    df_dupl = df[df.index.duplicated(keep=False)]
    return df_dupl
