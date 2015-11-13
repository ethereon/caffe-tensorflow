# Caffe to TensorFlow

This is a proof-of-concept that shows how to use pretrained Caffe models in TensorFlow.

This demo imports the [VGG-16 network](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) network.

Verified on the ILSVRC2012 validation set, on which it achieves a top 5 accuracy of 86.64%.

## Notes

- It appears that Caffe and TensorFlow cannot be concurrently invoked (CUDA conflicts - even with `set_mode_cpu`). This makes it a two-stage process: first extract the parameters with `kaffe.py`, then import it into TensorFlow.

- Caffe is not strictly required. However, the fallback uses the pure Python-based implementation of protobuf, which is astoundingly slow (~1.5 minutes to parse the VGG16 parameters). The experimental CPP protobuf backend doesn't particularly help here, since it runs into the file size limit (Caffe gets around this by overriding this limit in C++). A cleaner solution here would be to implement the loader as a C++ module.

- This implementation forces the padding to `SAME`. This may not necessarily be valid for your model, and not all Caffe paddings can be mapped to TensorFlow. See [Pete Warden's comments](https://github.com/ethereon/caffe-tensorflow/issues/3) on this issue for more details.
