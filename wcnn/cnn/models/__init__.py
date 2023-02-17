import sys

from . import alexnet, resnet

from .alexnet import AlexNet, ZhangAlexNet, WptAlexNet, \
                     DtCwptAlexNet, DtCwptZhangAlexNet
from .resnet import resnet18, zhang_resnet18, zou_resnet18, \
                    dtcwpt_resnet18, dtcwpt_zhang_resnet18, dtcwpt_zou_resnet18, \
                    resnet34, zhang_resnet34, zou_resnet34, \
                    dtcwpt_resnet34, dtcwpt_zhang_resnet34, dtcwpt_zou_resnet34, \
                    resnet50, zhang_resnet50, zou_resnet50, \
                    dtcwpt_resnet50, dtcwpt_zhang_resnet50, dtcwpt_zou_resnet50, \
                    resnet101, zhang_resnet101, zou_resnet101, \
                    dtcwpt_resnet101, dtcwpt_zhang_resnet101, dtcwpt_zou_resnet101


ARCH_LIST = sorted(
    name for name, obj in sys.modules[__name__].__dict__.items()
    if not name.startswith("__")
    and callable(obj)
)

def configs(arch):

    if "AlexNet" in arch:
        if "DtCwpt" in arch:
            out = alexnet.CONFIG_DICT_DTCWPTALEXNET.keys()
        elif "Wpt" in arch:
            raise NotImplementedError
        else:
            out = alexnet.CONFIG_DICT_STDALEXNET.keys()
    elif "resnet" in arch:
        if "dtcwpt" in arch:
            out = resnet.CONFIG_DICT_DTCWPTRESNET.keys()
        else:
            out = resnet.CONFIG_DICT_STDRESNET.keys()
    else:
        raise ValueError("Invalid architecture name")

    return out
