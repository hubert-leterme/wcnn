"""
Classes
-------
WaveCnnClassifier (inherits from sklearn.base.BaseEstimator)

Global variables
----------------
LR_SCHEDULER_DICT (dict)
    Organized as follows:
    - keys (str): learning rate scheduler names, which are possible values for
        the `lr_scheduler` constructor parameter of classifier.WaveCnnClassifier.
    - values (tuple): tuples containing:
        (a) PyTorch constructor learning rate schedulers;
        (b) dictionary of default arguments for the specified scheduler.

"""
from .classifier_cnn import WaveCnnClassifier
from .classifier_toolbox import LIST_OF_KWARGS_LOAD
