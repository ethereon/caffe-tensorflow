#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import examples

class ImageNet(object):
    def __init__(self, val_path, data_path, model):
        gt_lines = open(val_path).readlines()
        gt_pairs = [line.split() for line in gt_lines]
        self.image_paths = [os.path.join(data_path, p[0]) for p in gt_pairs]
        self.labels = np.array([int(p[1]) for p in gt_pairs])
        self.model = model    
        self.mean = np.array([104., 117., 124.])

    def read_image(self, path):
        img = cv2.imread(path)
        h, w, c = np.shape(img)
        scale_size = self.model.scale_size
        crop_size = self.model.crop_size
        assert c==3
        if self.model.isotropic:
            aspect = float(w)/h
            if w<h:
                resize_to = (scale_size, int((1.0/aspect)*scale_size))
            else:
                resize_to = (int(aspect*scale_size), scale_size)
        else:
            resize_to = (scale_size, scale_size)        
        img = cv2.resize(img, resize_to)
        img = img.astype(np.float32)
        img -= self.mean
        h, w, c = img.shape
        ho, wo = ((h-crop_size)/2, (w-crop_size)/2)
        img = img[ho:ho+crop_size, wo:wo+crop_size, :]
        img = img[None, ...]
        return img

    def batches(self, n):
        for i in xrange(0, len(self.image_paths), n):
            images = np.concatenate(map(self.read_image, self.image_paths[i:i+n]), axis=0)
            labels = self.labels[i:i+n]
            yield (images, labels)

    def __len__(self):
        return len(self.labels)

def test_imagenet(model, data_path, val_path, images_path, top_k=5):
    test_data   = tf.placeholder(tf.float32, shape=(model.batch_size, model.crop_size, model.crop_size, model.channels))
    test_labels = tf.placeholder(tf.int32, shape=(model.batch_size,))
    net         = model.net_class({'data':test_data})
    probs       = net.get_output()
    top_k_op    = tf.nn.in_top_k(probs, test_labels, top_k)
    imagenet    = ImageNet(val_path, images_path, model)
    correct     = 0
    count       = 0
    total       = len(imagenet)
    with tf.Session() as sesh:
        net.load(data_path, sesh)
        for idx, (images, labels) in enumerate(imagenet.batches(model.batch_size)):
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
        for idx, model in enumerate(examples.MODELS):
            print('%s: %s'%(idx, model))
        exit(-1)  
    model = examples.MODELS[model_index]
    print('Using model: %s'%(model))
    test_imagenet(model, *args[:3])

if __name__ == '__main__':
    main()
