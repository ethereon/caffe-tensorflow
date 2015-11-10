import os
import sys
import cv2
import kaffe
import numpy as np
import tensorflow as tf
from vgg import VGG16

BATCH_SIZE      = 20
IMAGE_SIZE      = 224
NUM_CHANNELS    = 3
NUM_LABELS      = 1000

class ImageNet(object):
    def __init__(self, val_path, data_path, mean=None, dim=IMAGE_SIZE):
        gt_lines = open(val_path).readlines()
        gt_pairs = [line.split() for line in gt_lines]
        self.image_paths = [os.path.join(data_path, p[0]) for p in gt_pairs]
        self.labels = np.array([int(p[1]) for p in gt_pairs])
        self.dim = dim
        if mean is None:
            self.mean = np.array([103.939, 116.779, 123.68])

    def read_image(self, path):        
        img = cv2.imread(path)
        img = cv2.resize(img, (self.dim, self.dim))
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

def test_imagenet(params_path, val_path, data_path, top_k=5):
    test_data   = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    test_labels = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))
    logits      = VGG16(test_data).get_output()
    top_k_op    = tf.nn.in_top_k(logits, test_labels, top_k)    
    imagenet    = ImageNet(val_path, data_path)
    correct     = 0    
    with tf.Session() as sesh:        
        caffe_loader = kaffe.CaffeLoader(params_path)
        caffe_loader.load(sesh)
        for idx, (images, labels) in enumerate(imagenet.batches(BATCH_SIZE)):
            correct += np.sum(sesh.run(top_k_op, feed_dict={test_data:images, test_labels:labels}))
            print('%d/%d'%((idx+1)*BATCH_SIZE, len(imagenet)))

    print('Top %s Accuracy: %s'%(top_k, float(correct)/len(imagenet)))

def main():
    args = sys.argv[1:]
    if len(args)!=3:
        print('usage: %s net.params imagenet-val.txt imagenet-data-dir'%os.path.basename(__file__))
        exit(-1)
    test_imagenet(*args)

if __name__ == '__main__':
    main()