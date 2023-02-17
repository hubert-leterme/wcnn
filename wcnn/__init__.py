import os
import sys
import yaml
import warnings
import torch

# First, check if the config file wcnn_config.yml is in the current directory.
# If not in there, check in "~/.config/".
CONFIG_DIRLIST = [os.getcwd(), os.path.expanduser("~/.config")]

CONFIGFILE = None
_iter_config_dirlist = iter(CONFIG_DIRLIST)
while CONFIGFILE is None:
    try:
        configdir = next(_iter_config_dirlist)
    except StopIteration:
        pass
    else:
        test_path = os.path.join(configdir, "wcnn_config.yml")
        if os.path.isfile(test_path):
            CONFIGFILE = test_path

if CONFIGFILE is None:
    warnings.warn("No configuration file provided.")
    CONFIGDICT = None
else:
    with open(CONFIGFILE, 'r') as stream:
        CONFIGDICT = yaml.safe_load(stream)

# Append directories to sys.path
if CONFIGDICT is not None:
    for dep in CONFIGDICT["external_repositories"]:
        path = dep["root"]
        if path is None:
            msg = "No path provided for {}.\n".format(dep["name"])
            msg += "URL to the Git repository: {}\n".format(dep["url"])
            msg += "Commit number: {}".format(dep["commit"])
            warnings.warn(msg)
        else:
            sys.path.append(os.path.expanduser(path))

# GPU computing
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA else "cpu")
N_GPUS = torch.cuda.device_count() if CUDA else None
