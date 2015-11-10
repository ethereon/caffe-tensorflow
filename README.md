# Caffe to TensorFlow

This is a proof-of-concept that shows how to use pretrained Caffe models in TensorFlow.

This demo imports the [VGG-16 network](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) network.

Verified on the ILSVRC2012 validation set, on which it achieves a top 5 accuracy of 86.64%.

A minor quirk: you need to run `kaffe.py` to extract the parameters first, since it appears that Caffe and TensorFlow don't play nice (CUDA conflicts). This persists even with `set_mode_cpu`. However, for future work, there's really no need to depend on Caffe here: just the protocol buffers should suffice.

