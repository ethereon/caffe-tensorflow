# Caffe to TensorFlow

Convert [Caffe](https://github.com/BVLC/caffe/) models to [TensorFlow](https://github.com/tensorflow/tensorflow).

## Usage

Run `convert.py` to convert an existing Caffe model to TensorFlow.

Make sure you're using the latest Caffe format (see the notes section for more info).

The output consists of two files:

1. A data file (in NumPy's native format) containing the model's learned parameters.
2. A Python class that constructs the model's graph.

### Example

Convert the model:

    ./convert.py deploy.prototxt net.caffemodel mynet.npy mynet.py

Inference:

```python
# Import the converted model's class
from mynet import MyNet

# Create an instance, passing in the input data
net = MyNet({'data':my_input_data})

with tf.Session() as sesh:
    # Load the data
    net.load('mynet.npy', sesh)
    # Forward pass
    output = sesh.run(net.get_output(), ...)
```

See `test.py` for a functioning example. It verifies the sample models (under `examples/`) against the ImageNet validation set.

## Verification

The following converted models have been verified on the ILSVRC2012 validation set.

| Model                                          | Top 5 Accuracy |
|:-----------------------------------------------|---------------:|
| [VGG 16](http://arxiv.org/abs/1409.1556)       |         89.88% |
| [GoogLeNet](http://arxiv.org/abs/1409.4842)    |         89.06% |

## Notes

- Only the new Caffe model format is supported. If you have an old model, use the `upgrade_net_proto_text` and `upgrade_net_proto_binary` tools that ship with Caffe to upgrade them first. Also make sure you're using a fairly recent version of Caffe.

- It appears that Caffe and TensorFlow cannot be concurrently invoked (CUDA conflicts - even with `set_mode_cpu`). This makes it a two-stage process: first extract the parameters with `convert.py`, then import it into TensorFlow.

- Caffe is not strictly required. If PyCaffe is found in your `PYTHONPATH`, it will be used. Otherwise, a fallback will be used. However, the fallback uses the pure Python-based implementation of protobuf, which is astoundingly slow (~1.5 minutes to parse the VGG16 parameters). The experimental CPP protobuf backend doesn't particularly help here, since it runs into the file size limit (Caffe gets around this by overriding this limit in C++). A cleaner solution here would be to implement the loader as a C++ module.

- Only a subset of Caffe layers and accompanying parameters are currently supported. 

- Not all Caffe models can be converted to TensorFlow. For instance, Caffe supports arbitrary padding whereas TensorFlow's support is currently restricted to `SAME` and `VALID`.

- The border values are handled differently by Caffe and TensorFlow. However, these don't appear to affect things too much.

- Image rescaling can affect the ILSVRC2012 top 5 accuracy listed above slightly. VGG16 expects isotropic rescaling (anisotropic reduces accuracy to 88.45%) whereas BVLC's implementation of GoogLeNet expects anisotropic (isotropic reduces accuracy to 87.7%).

- The support class `kaffe.tensorflow.Network` has no internal dependencies. It can be safely extracted and deployed without the rest of this library.
