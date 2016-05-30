# Caffe to TensorFlow

Convert [Caffe](https://github.com/BVLC/caffe/) models to [TensorFlow](https://github.com/tensorflow/tensorflow).

## Usage

Run `convert.py` to convert an existing Caffe model to TensorFlow.

Make sure you're using the latest Caffe format (see the notes section for more info).

The output consists of two files:

1. A data file (in NumPy's native format) containing the model's learned parameters.
2. A Python class that constructs the model's graph.

### Examples

See the [examples](examples/) folder for more details.

## Verification

The following converted models have been verified on the ILSVRC2012 validation set using
[validate.py](examples/imagenet/validate.py).

| Model                                                 | Top 5 Accuracy |
|:------------------------------------------------------|---------------:|
| [ResNet 152](http://arxiv.org/abs/1512.03385)         |         92.92% |
| [ResNet 101](http://arxiv.org/abs/1512.03385)         |         92.63% |
| [ResNet 50](http://arxiv.org/abs/1512.03385)          |         92.02% |
| [VGG 16](http://arxiv.org/abs/1409.1556)              |         89.88% |
| [GoogLeNet](http://arxiv.org/abs/1409.4842)           |         89.06% |
| [Network in Network](http://arxiv.org/abs/1312.4400)  |         81.21% |
| [CaffeNet](http://arxiv.org/abs/1408.5093)            |         79.93% |
| [AlexNet](http://goo.gl/3BilWd)                       |         79.84% |

## Notes

- Only the new Caffe model format is supported. If you have an old model, use the `upgrade_net_proto_text` and `upgrade_net_proto_binary` tools that ship with Caffe to upgrade them first. Also make sure you're using a fairly recent version of Caffe.

- It appears that Caffe and TensorFlow cannot be concurrently invoked (CUDA conflicts - even with `set_mode_cpu`). This makes it a two-stage process: first extract the parameters with `convert.py`, then import it into TensorFlow.

- Caffe is not strictly required. If PyCaffe is found in your `PYTHONPATH`, and the `USE_PYCAFFE` environment variable is set, it will be used. Otherwise, a fallback will be used. However, the fallback uses the pure Python-based implementation of protobuf, which is astoundingly slow (~1.5 minutes to parse the VGG16 parameters). The experimental CPP protobuf backend doesn't particularly help here, since it runs into the file size limit (Caffe gets around this by overriding this limit in C++). A cleaner solution here would be to implement the loader as a C++ module.

- Only a subset of Caffe layers and accompanying parameters are currently supported.

- Not all Caffe models can be converted to TensorFlow. For instance, Caffe supports arbitrary padding whereas TensorFlow's support is currently restricted to `SAME` and `VALID`.

- The border values are handled differently by Caffe and TensorFlow. However, these don't appear to affect things too much.

- Image rescaling can affect the ILSVRC2012 top 5 accuracy listed above slightly. VGG16 expects isotropic rescaling (anisotropic reduces accuracy to 88.45%) whereas BVLC's implementation of GoogLeNet expects anisotropic (isotropic reduces accuracy to 87.7%).

- The support class `kaffe.tensorflow.Network` has no internal dependencies. It can be safely extracted and deployed without the rest of this library.

- The ResNet model uses 1x1 convolutions with a stride of 2. This is currently only supported in the master branch of TensorFlow (the latest release at time of writing being v0.8.0, which does not support it).
