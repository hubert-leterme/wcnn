import math

def pop_attrs(obj, *attrs):
    """
    Set a list of attribute to None and return the old values.

    """
    out = []
    for attr in attrs:
        out.append(getattr(obj, attr))
        setattr(obj, attr, None)

    return out


def get_error(score):

    if score is None:
        out = None
    elif isinstance(score, float):
        out = 1 - score
    elif isinstance(score, dict):
        out = {}
        for N in score.keys():
            out[N] = 1 - score[N]
    else:
        raise NotImplementedError

    return out


def extract_dict(list_of_keys, inp, discard_none=False, pop=True):

    out = {}
    for key in list_of_keys:
        if key in inp:
            if pop:
                val = inp.pop(key)
            else:
                val = inp.get(key)
            if not (discard_none and val is None):
                out[key] = val

    return out


def isqrt(n):

    # Function math.isqrt exists in the standard library since Python 3.8
    try:
        out = math.isqrt(n)

    # For backward compatibility (Newton's method)
    # From https://stackoverflow.com/questions/15390807/integer-square-root-in-python
    except AttributeError:
        out = n
        val = (out + 1) // 2
        while val < out:
            out = val
            val = (out + n // out) // 2

    return out


class InfoPrinter(object):

    def __init__(self, distributed, gpu, verbose=False):
        self.distributed = distributed
        self.gpu = gpu
        self.verbose=verbose

    def __call__(self, *args, always=False, all_gpus=False):
        if not self.distributed:
            if always or self.verbose:
                print(*args)
        else:
            if always:
                if not all_gpus:
                    if self.gpu == 0:
                        print(*args)
                else:
                    self._print_with_prefix(*args)
            elif self.verbose: # Print info from all GPUs
                self._print_with_prefix(*args)

    def _print_with_prefix(self, *args):
        prefix = "[GPU {}]".format(self.gpu)
        print(prefix, *args)
