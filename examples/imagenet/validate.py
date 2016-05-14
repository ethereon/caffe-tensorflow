#!/usr/bin/env python
'''Validates a converted ImageNet model against the ILSVRC12 validation set.'''

import argparse
import numpy as np
import tensorflow as tf
import os.path as osp

import models
import dataset


def load_model(name):
    '''Creates and returns an instance of the model given its class name.
    The created model has a single placeholder node for feeding images.
    '''
    # Find the model class from its name
    all_models = models.get_models()
    lut = {model.__name__: model for model in all_models}
    if name not in lut:
        print('Invalid model index. Options are:')
        # Display a list of valid model names
        for model in all_models:
            print('\t* {}'.format(model.__name__))
        return None
    NetClass = lut[name]

    # Create a placeholder for the input image
    spec = models.get_data_spec(model_class=NetClass)
    data_node = tf.placeholder(tf.float32,
                               shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct and return the model
    return NetClass({'data': data_node})


def validate(net, model_path, images, top_k=5):
    '''Compute the top_k classification accuracy for the given network and images.'''
    # Get the data specifications for given network
    spec = models.get_data_spec(model_instance=net)
    # Get the input node for feeding in the images
    input_node = net.inputs['data']
    # Create a placeholder for the ground truth labels
    label_node = tf.placeholder(tf.int32, shape=(spec.batch_size,))
    # Get the output of the network (class probabilities)
    probs = net.get_output()
    # Create a top_k accuracy node
    top_k_op = tf.nn.in_top_k(probs, label_node, top_k)
    # The number of images processed
    count = 0
    # The number of correctly classified images
    correct = 0
    # The total number of images
    total = len(images)
    with tf.Session() as sesh:
        # Load the converted parameters
        net.load(model_path, sesh)
        # Iterate over and classify mini-batches
        for idx, (images, labels) in enumerate(images.batches(spec.batch_size)):
            correct += np.sum(sesh.run(top_k_op,
                                       feed_dict={input_node: images.eval(),
                                                  label_node: labels}))
            count += images.get_shape()[0].value
            cur_accuracy = float(correct) * 100 / count
            print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
    print('Top {} Accuracy: {}'.format(top_k, float(correct) / total))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to the converted model parameters (.npy)')
    parser.add_argument('val_gt', help='Path to validation set ground truth (.txt)')
    parser.add_argument('imagenet_data_dir', help='ImageNet validation set images directory path')
    parser.add_argument('--model', default='GoogleNet', help='The name of the model to evaluate')
    args = parser.parse_args()

    # Load the network
    net = load_model(args.model)

    # Load the dataset
    data_spec = models.get_data_spec(model_instance=net)
    images = dataset.ImageNet(args.val_gt, args.imagenet_data_dir, data_spec)

    # Evaluate its performance on the ILSVRC12 validation set
    validate(net, args.model_path, images)


if __name__ == '__main__':
    main()
