# <b>WCNN</b>

**From CNNs to Shift-Invariant Twin Wavelet Models** ([arXiv:2212.00394](https://arxiv.org/abs/2212.00394))

*Hubert Leterme, Kévin Polisano, Valérie Perrier, Karteek Alahari*

<sub>Univ. Grenoble Alpes, CNRS, Inria, Grenoble INP, LJK, 38000 Grenoble, France</sub>

## Requirements and Settings

You can create a conda environment to reproduce our results in similar conditions.

```bash
conda env create -f condaenv.yml
conda activate wcnn
```

**Before running the scripts or using the library,** update the configuration file provided at the root of this repository, named `wcnn_config.yml`, as follows.

#### `datasets`

Specify the root directories of the datasets.

#### `classifiers`

The repository includes:
- a file `./classif_db.csv`, which lists available classifiers and the corresponding scripts, for reproducibility;
- a directory `./classif/`, which contains the pickled training and evaluation metrics (`<filename>.classif`) and PyTorch state dictionaries (`<filename>.tar`) for each classifier.

The paths are already set to the correct locations. However, if you want to place these assets in another directory, you need to update `path_to_csv` and `path_to_classif`.

#### `external_repositories`

These are GitHub repositories not handled by pip. Therefore, you must clone them yourself and specify their location in front of `root`. They will be automatically appended to `sys.path`.

#### `gpu_clusters`

If you trained models on a GPU cluster, you may need to download the pickled classifiers from the remote location. This can be handled automatically if accessed via SSH. For this to be possible, you must specify the host name (as `host1` or `host2`) as well as the location of the directory containing the classifiers (corresponding to `path_to_classif` in the remote configuration file).

## Loading an Existing Classifier

```python
from wcnn.classifier import WaveCnnClassifier
```
```python
# Get list of available classifiers
WaveCnnClassifier.list()
```
Here is a list of trained classifiers provided in this repository. **We only provided the training and evaluation metrics** (`<filename>.classif`), due to the storage limit. The files containing the PyTorch state dictionaries (`<filename>.tar`) can be retrieved by re-training the classifiers (see below).

| Name | Baseline | Model | Dataset |
| :--: | :---: | :-----------: | :-----: |
| `AI` | AlexNet | Standard | ImageNet |
| `AwyI` | AlexNet | $\mathbb R$-Max | ImageNet |
| `AwyI_mod` | AlexNet | $\mathbb C$-Mod | ImageNet |
| `AzhI` | AlexNet | Blur (`filter_size=4`) | ImageNet |
| `AzhwyI` | AlexNet | Blur-$\mathbb R$-Max + Blur | ImageNet |
| `AzhwyI_mod` | AlexNet | $\mathbb C$-Mod + Blur | ImageNet |
| `RI` | ResNet-34 | Standard | ImageNet |
| `RwyI` | ResNet-34 | $\mathbb R$-Max | ImageNet |
| `RwyI_mod` | ResNet-34 | $\mathbb C$-Mod | ImageNet |
| `RzhI` | ResNet-34 | Blur (`filter_size=4`) | ImageNet |
| `RzhwyI` | ResNet-34 | Blur-$\mathbb R$-Max + Blur | ImageNet |
| `RzhwyI_mod` | ResNet-34 | $\mathbb C$-Mod + Blur | ImageNet |
| `Rzof3I` | ResNet-34 | ABlur (`filter_size=3`) | ImageNet |
| `Rzof3wyI` | ResNet-34 | ABlur-$\mathbb R$-Max + ABlur | ImageNet |
| `Rzof3wyI_mod` | ResNet-34 | $\mathbb C$-Mod + ABlur | ImageNet |
| `R18C` | ResNet-18 | Standard | CIFAR10 |
| `R18wyC` | ResNet-18 | $\mathbb R$-Max | CIFAR10 |
| `R18wyC_mod` | ResNet-18 | $\mathbb C$-Mod | CIFAR10 |
| `R18zhf3C` | ResNet-18 | Blur (`filter_size=3`) | CIFAR10 |
| `R18zhf3wyC` | ResNet-18 | Blur-$\mathbb R$-Max + Blur | CIFAR10 |
| `R18zhf3wyC_mod` | ResNet-18 | $\mathbb C$-Mod + Blur | CIFAR10 |
| `R18zof3C` | ResNet-18 | ABlur (`filter_size=3`) | CIFAR10 |
| `R18zof3wyC` | ResNet-18 | ABlur-$\mathbb R$-Max + ABlur | CIFAR10 |
| `R18zof3wyC_mod` | ResNet-18 | $\mathbb C$-Mod + ABlur | CIFAR10 |

```python
# Load classifier (WAlexNet trained on ImageNet, 90 epochs)
classif = WaveCnnClassifier.load("AwyI", status="e90")
```

The status can be equal to `checkpoint` (last saved checkpoint from this classifier), `best` (checkpoint with the best validation score so far; disabled by default), or `e<nepochs>`, where `<nepochs>` denotes the corresponding training epoch.

The object `classif` inherits from `sklearn.base.BaseEstimator`. Here is a (non-comprehensive) list of attributes, updated during training or evaluation.

```python
# Trained PyTorch model
# If state dicts are unavailable, then the parameters will not be in their trained state.
net = classif.net_

# List of losses, for each minibatch of size classif.batch_size_train
losses = classif.losses_

# Dictionary of validation errors, for each training epoch
val_errs = classif.val_errs_

# Dictionary of evaluation scores (accuracy, KL divergence, mean flip rate...), computed after training
eval_scores = classif.eval_scores_
```

## Resulting Convolution Kernel

In a WCNN, the first convolution layer is replaced by a PyTorch module of type `wcnn.cnn.building_blocks.HybridConv2dWpt`, which includes a freely-trained layer of type `torch.nn.Conv2d`, and a wavelet block of type `wcnn.cnn.building_blocks.WptBlock`.

A module `HybridConv2dWpt` can be viewed as a conventional convolution layer with a reduced number of degrees of freedom. As such, it is possible to compute an "equivalent convolution kernel", as explained in the paper.

```python
# In a standard AlexNet, this module would be of type torch.nn.Conv2d
hybridconv = net.features[0]

# PyTorch tensor of size (64, 3, 72, 72)
ker = hybridconv.resulting_kernel
```

Note that the kernel size ($72 \times 72$) is much bigger than in a conventional AlexNet ($11 \times 11$). However, most of its energy lies in a much smaller region because the filter coefficients are fast decaying.

Besides, this resulting kernel is only used for visualization purpose. For computational reasons, the wavelet packet coefficients are actually computed with successive subsampled convolutions and linear combinations of feature maps.

## Evaluation Scores

The evaluation scores are stored in a nested dictionary (`classif.eval_scores_`) with the following structure:

```
- ds_name # "ImageNet", "CIFAR10"...
    - split # "val", "test", or other splits depending on the dataset
        - eval_mode # "onecrop", "tencrops", "shifts"...
            - metrics # "top1_5_accuracy_scores", "kldiv", "mfr"...
```

For reproducibility reasons, the scores are stored in an object of type `wcnn.classifier.classifier_toolbox.Score`, which also contains the recording date and time, and the commit number (if provided when computing the score).

## Training a Model

### AlexNet on ImageNet

```bash
#!/bin/bash

# Standard
python train.py AI -a AlexNet --lr 0.01

# RMax
python train.py AwyI -a AlexNet --config Ft32Y --lr 0.01 --get-gradients --has-l1loss --lambda-params 0. 0.0041 0.00032

# CMod
python train.py AwyI_mod -a DtCwptAlexNet --config Ft32Y_mod --lr 0.01 --get-gradients --has-l1loss --lambda-params 0. 0.0041 0.00032

# Blur (static blur pooling)
python train.py AzhI -a ZhangAlexNet --lr 0.01

# BlurRMax + Blur
python train.py AzhwyI -a DtCwptZhangAlexNet --config Ft32Y --lr 0.01 --get-gradients --has-l1loss --lambda-params 0. 0.0041 0.00032

# CMod + Blur
python train.py AzhwyI_mod -a DtCwptZhangAlexNet --config Ft32Y_mod --lr 0.01 --get-gradients --has-l1loss --lambda-params 0. 0.0041 0.00032
```

### ResNet on ImageNet

```bash
#!/bin/bash

# Standard
python train.py RI -a resnet34

# RMax
python train.py RwyI -a dtcwpt_resnet34 --config Ft40YEx

# CMod
python train.py RwyI_mod -a dtcwpt_resnet34 --config Ft40YEx_mod

# Blur (static blur pooling)
python train.py RzhI -a zhang_resnet34

# BlurRMax + Blur
python train.py RzhwyI -a dtcwpt_zhang_resnet34 --config Ft40YEx

# CMod + Blur
python train.py RzhwyI_mod -a dtcwpt_zhang_resnet34 --config Ft40YEx_mod

# ABlur (adaptive blur pooling)
python train.py Rzof3I -a zou_resnet34 --config Bf3 -ba 2

# ABlurRMax + ABlur
python train.py Rzof3wyI -a dtcwpt_zou_resnet34 --config Ft40YBf3Ex -ba 2

# CMod + ABlur
python train.py Rzof3wyI_mod -a dtcwpt_zou_resnet34 --config Ft40YBf3Ex_mod -ba 2
```

### ResNet on CIFAR10

```bash
#!/bin/bash

# To use ResNet-34, 50 or 101 instead of 18, simply replace the number of layers in the following scripts.

# Standard
python train.py R18C -a resnet18 -ds CIFAR10 --epochs 300 --lr-scheduler StepLR100

# RMax
python train.py R18wyC -a dtcwpt_resnet18 --config Ft40YEx -ds CIFAR10 --epochs 300 --lr-scheduler StepLR100

# CMod
python train.py R18wyC_mod -a dtcwpt_resnet18 --config Ft40YEx_mod -ds CIFAR10 --epochs 300 --lr-scheduler StepLR100

# Blur (static blur pooling)
python train.py R18zhf3C -a zhang_resnet18 --config Bf3 -ds CIFAR10 --epochs 300 --lr-scheduler StepLR100

# BlurRMax + Blur
python train.py R18zhf3wyC -a dtcwpt_zhang_resnet18 --config Ft40YBf3Ex -ds CIFAR10 --epochs 300 --lr-scheduler StepLR100

# CMod + Blur
python train.py R18zhf3wyC_mod -a dtcwpt_zhang_resnet18 --config Ft40YBf3Ex_mod -ds CIFAR10 --epochs 300 --lr-scheduler StepLR100

# ABlur (adaptive blur pooling)
python train.py R18zof3C -a zou_resnet18 --config Bf3 -ds CIFAR10 --epochs 300 --lr-scheduler StepLR100

# ABlurRMax + ABlur
python train.py R18zof3wyC -a dtcwpt_zou_resnet18 --config Ft40YBf3Ex -ds CIFAR10 --epochs 300 --lr-scheduler StepLR100

# CMod + ABlur
python train.py R18zof3wyC_mod -a dtcwpt_zou_resnet18 --config Ft40YBf3Ex_mod -ds CIFAR10 --epochs 300 --lr-scheduler StepLR100
```

### Model configurations (argument `--config`) 

```python
from wcnn.cnn import models

# List of available architectures
arch_list = models.ARCH_LIST

# List of available configurations for a given architecture
configs = models.configs("dtcwpt_resnet34")
```

The configurations are named according to the following nomenclature.

| Prototype | Meaning |
| :-------: | :-----: |
| `Ft32` | $32$ freely-trained channels |
| `Y` | Color mixing before wavelet transform |
| `Bf3` | Blurring filter of size $3$ (otherwise, default $= 4$) |
| `Ex` | Exclude edge filters (see remark below) |
| `_mod` | Replace $\mathbb R$-Max (standalone or with blur pooling) by standalone $\mathbb C$-Mod |

### Remarks

- The first (positional) argument of the Python script `train.py` (e.g., `AI`) is used as an identifier for the classifier. You can use any string, unless a classifier with the same name already exists in `classif_db.csv`.

- If a script stops before the end, simply re-launch it to resume training from the last checkpoint. To resume training from a specific epoch, add argument `-r &epoch`, where `&epoch` denotes the requested epoch number.

- To avoid CUDA out-of-memory errors, in models with adaptive blur pooling, we set option `-ba 2` (batch accumulation before updating weights), following [Zou et al.](https://github.com/MaureenZOU/Adaptive-anti-Aliasing) See also [this discussion on the PyTorch forum](https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20?u=alband).

- Following [the standard procedures](https://github.com/pytorch/examples/tree/main/imagenet), in AlexNet-based models, the initial learning rate is set to $0.01$ (the default value is $0.1$).

- In ResNet-based models, the $14$ filters with maximal frequency ("edge" filters) have been manually discarded. These excluded filters have indeed poorly-defined orientations, and often take the appearance of a checkerboard. To use these filters nevertheless, simply replace (for instance) `--config Ft40YEx` by `--config Ft40Y` in the above scripts.

- There is no $l^1/l^\infty$-regularization for ResNet-based models. This is because the number of manually selected wavelet packet filters ($16$) is sufficiently small comparatively to the available number of output channels ($24$ assigned to the wavelet block). Therefore, all filters are mapped to at least one output channel. If `--config Ft40YEx` is replaced by `--config Ft40Y` (see above), then regularization is recommended. In this case, add parameters `--get-gradients` (optional), `--has-l1loss` and `--lambda-params 0. 0.005`. For instance:

```bash
# RMax
python train.py Rwy2I -a dtcwpt_resnet34 --config Ft40Y --get-gradients --has-l1loss --lambda-params 0. 0.005
```

- For ImageNet training, the blurring filter size is respectively set to $4$ (static) and $3$ (adaptive), which are the default values in the corresponding libraries. However, for CIFAR10, the blurring filter size is set to $3$ in any case (otherwise, errors occur due to the small size of input images).

## Model Evaluation

Examples with WAlexNet trained on ImageNet ($90$ epochs). The following scores are computed on the "test" dataset, which is actually the official "validation" set provided by ImageNet ($50\,000$ images). Note that, during training, a subset of the training set ($100\,000$ images) is used for validation.

### Accuracy Scores

```bash
#!/bin/bash

# One crop
python eval.py AwyI -s e90 -em onecrop

# Ten crops
python eval.py AwyI -s e90 -b 32 -em tencrops --verbose-print-batch 500
```

### Shift Invariance

The examples are given for `horizontal` shifts (can also be set to `vertical` or `diagonal`).

```bash
#!/bin/bash

# KL divergence
python eval.py AwyI -s e90 -b 16 -em shifts --max-shift 8 --pixel-divide 2 --shift-direction horizontal --metrics kldiv --verbose-print-batch 256

# Mean flip rate (mFR)
python eval.py AwyI -s e90 -b 16 -em shifts --max-shift 8 --pixel-divide 1 --shift-direction horizontal --max-step-size 8 --verbose-print-batch 256
```

#### Notes for mFR

- `--pixel-divide 1`: input shifts are done by steps of $1$ pixel ($0.5$ for KL divergence);
- `--max-step-size 8`: compute mFR up to 8-pixel shifts. For models trained with CIFAR-10, use `--max-step-size 4` instead.

## License

Copyright 2023 Hubert Leterme

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Citation

H. Leterme, K. Polisano, V. Perrier, and K. Alahari, “From CNNs to Shift-Invariant Twin Wavelet Models,” arXiv:2212.00394, Dec. 2022.
