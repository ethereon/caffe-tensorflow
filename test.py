#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

import examples
from kaffe.tensorflow import TensorFlowLoader

BATCH_SIZE      = 25
PRE_CROP_SIZE   = 256
IMAGE_SIZE      = 224
NUM_CHANNELS    = 3
NUM_LABELS      = 1000

class ImageNet(object):
    def __init__(self, val_path, data_path, mean=None):
        gt_lines = open(val_path).readlines()
        gt_pairs = [line.split() for line in gt_lines]
        self.image_paths = [os.path.join(data_path, p[0]) for p in gt_pairs]
        self.labels = np.array([int(p[1]) for p in gt_pairs])
        if mean is None:
            self.mean = np.array([103.939, 116.779, 123.68])

    def read_image(self, path):
        img = cv2.imread(path)
        h, w, c = np.shape(img)
        assert c==3
        aspect = float(w)/h
        if w<h:
            resize_to = (PRE_CROP_SIZE, int((1.0/aspect)*PRE_CROP_SIZE))
        else:
            resize_to = (int(aspect*PRE_CROP_SIZE), PRE_CROP_SIZE)
        img = cv2.resize(img, resize_to)
        h, w, c = img.shape
        delta = IMAGE_SIZE/2
        img = img[(h/2)-delta:(h/2)+delta, (w/2)-delta:(w/2)+delta, :]
        img = img.astype(np.float32)
        img -= self.mean
        img = img[None, ...]
        return img

    def batches(self, n):
        for i in xrange(0, len(self.image_paths), n):
            images = np.concatenate(map(self.read_image, self.image_paths[i:i+n]), axis=0)
            labels = self.labels[i:i+n]
            yield (images, labels)

    def __len__(self):
        return len(self.labels)

def test_imagenet(Net, params_path, val_path, data_path, top_k=5):
    test_data   = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    test_labels = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))
    probs       = Net({'data':test_data}).get_output()
    top_k_op    = tf.nn.in_top_k(probs, test_labels, top_k)
    imagenet    = ImageNet(val_path, data_path)
    correct     = 0
    count       = 0
    total       = len(imagenet)
    with tf.Session() as sesh:
        caffe_loader = TensorFlowLoader(params_path)
        caffe_loader.load(sesh)
        for idx, (images, labels) in enumerate(imagenet.batches(BATCH_SIZE)):
            correct += np.sum(sesh.run(top_k_op, feed_dict={test_data:images, test_labels:labels}))
            count += len(images)
            cur_accuracy = float(correct)*100/count
            print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
    print('Top %s Accuracy: %s'%(top_k, float(correct)/total))

def main():
    args = sys.argv[1:]
    if len(args) not in (3, 4):
        print('usage: %s net.params imagenet-val.txt imagenet-data-dir [model-index=0]'%os.path.basename(__file__))
        exit(-1)
    model_index = 0 if len(args)==3 else int(args[3])
    if model_index>=len(examples.MODELS):
        print('Invalid model index. Options are:')
        for idx, klass in enumerate(examples.MODELS):
            print('%s: %s'%(idx, klass.__name__))
        exit(-1)  
    Net = examples.MODELS[model_index]
    print('Using model: %s'%(Net.__name__))
    test_imagenet(Net, *args[:3])

if __name__ == '__main__':
    main()
