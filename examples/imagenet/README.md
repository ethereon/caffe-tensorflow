# ImageNet Examples

This folder contains two examples that demonstrate how to use converted networks for
image classification. Also included are sample converted models and helper scripts.

## 1. Image Classification

`classify.py` uses a GoogleNet trained on ImageNet, converted to TensorFlow, for classifying images.

The architecture used is defined in `models/googlenet.py` (which was auto-generated). You will need
to download and convert the weights from Caffe to run the example. The download link for the
corresponding weights can be found in Caffe's `models/bvlc_googlenet/` folder.

You can run this example like so:

    $ ./classify.py /path/to/googlenet.npy ~/pics/kitty.png ~/pics/woof.jpg

You should expect to see an output similar to this:

    Image                Classified As                  Confidence
    ----------------------------------------------------------------------
    kitty.png            Persian cat                    99.75 %
    woof.jpg             Bernese mountain dog           82.02 %


## 2. ImageNet Validation

`validate.py` evaluates a converted model against the ImageNet (ILSVRC12) validation set. To run
this script, you will need a copy of the ImageNet validation set. You can run it as follows:

    $ ./validate.py alexnet.npy val.txt imagenet-val/ --model AlexNet

The validation results specified in the main readme were generated using this script.

## Helper Scripts

In addition to the examples above, this folder includes a few additional files:

- `dataset.py` : helper script for loading, pre-processing, and iterating over images
- `models/` : contains converted models (auto-generated)
- `models/helper.py` : describes how the data should be preprocessed for each model
