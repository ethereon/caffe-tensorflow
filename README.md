# Caffe to TensorFlow

This is a proof-of-concept that shows how to use pretrained Caffe models in TensorFlow.

This demo imports the [VGG-16 network](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) network.

Verified on the ILSVRC2012 validation set, on which it achieves a top 5 accuracy of 89.88%.

## Notes

- Only the new Caffe model format is supported. If you have an old model, use the `upgrade_net_proto_text` and `upgrade_net_proto_binary` tools that ship with Caffe to upgrade them first.

- It appears that Caffe and TensorFlow cannot be concurrently invoked (CUDA conflicts - even with `set_mode_cpu`). This makes it a two-stage process: first extract the parameters with `convert.py`, then import it into TensorFlow.

- Caffe is not strictly required. However, the fallback uses the pure Python-based implementation of protobuf, which is astoundingly slow (~1.5 minutes to parse the VGG16 parameters). The experimental CPP protobuf backend doesn't particularly help here, since it runs into the file size limit (Caffe gets around this by overriding this limit in C++). A cleaner solution here would be to implement the loader as a C++ module.

- Not all Caffe paddings can be mapped to TensorFlow. See [the comments on this issue](https://github.com/ethereon/caffe-tensorflow/issues/3) for more details.
