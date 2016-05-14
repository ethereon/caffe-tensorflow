### LeNet Example

_Thanks to @Russell91 for this example_

This example showns you how to finetune code from the [Caffe MNIST tutorial](http://caffe.berkeleyvision.org/gathered/examples/mnist.html) using Tensorflow.
First, you can convert a prototxt model to tensorflow code:

    $ ./convert.py examples/mnist/lenet.prototxt --code-output-path=mynet.py

This produces tensorflow code for the LeNet network in `mynet.py`. The code can be imported as described below in the Inference section. Caffe-tensorflow also lets you convert `.caffemodel` weight files to `.npy` files that can be directly loaded from tensorflow:

    $ ./convert.py examples/mnist/lenet.prototxt --caffemodel examples/mnist/lenet_iter_10000.caffemodel --data-output-path=mynet.npy

The above command will generate a weight file named `mynet.npy`.

#### Inference:

Once you have generated both the code weight files for LeNet, you can finetune LeNet using tensorflow with

    $ ./examples/mnist/finetune_mnist.py

At a high level, `finetune_mnist.py` works as follows:

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
