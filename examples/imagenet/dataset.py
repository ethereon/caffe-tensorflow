'''Utility functions and classes for handling image datasets.'''

import os.path as osp
import numpy as np
import tensorflow as tf


def read_image(path, to_bgr=True):
    '''Returns the image at the given path as a tensor.'''
    # Read the file
    file_data = tf.read_file(path)
    # Figure out the image format from the extension
    ext = osp.splitext(path)[-1].lower()
    if ext == '.png':
        decoder = tf.image.decode_png
    elif ext in ('.jpg', '.jpeg'):
        decoder = tf.image.decode_jpeg
    else:
        raise ValueError('Unsupported image extension: {}'.format(ext))
    img = decoder(file_data, channels=3)
    if to_bgr:
        # Convert from RGB channel ordering to BGR
        # This matches, for instance, how OpenCV orders the channels.
        img = tf.reverse(img, [False, False, True])
    return img


def _load_image(path, scale, isotropic, crop, mean):
    '''Loads and pre-processes the image at the given path.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, a central crop of this size is taken.
    mean  : Subtracted from the image
    '''
    # Read in the image
    img = read_image(path)
    # Rescale
    if isotropic:
        scale = scale / float(min(w, h))
        new_height, new_width = (h * scale, w * scale)
    else:
        new_height, new_width = (scale, scale)
    img = tf.image.resize_images(img, new_height, new_width)
    # Crop
    img = tf.image.crop_to_bounding_box(img, (new_height - crop) / 2,
                                        (new_width - crop) / 2, crop, crop)
    # Mean subtraction
    return tf.to_float(img) - mean


def load_image(path, spec):
    '''Load a single image, processed based on the given spec.'''
    return _load_image(path=path,
                       scale=spec.scale_size,
                       isotropic=spec.isotropic,
                       crop=spec.crop_size,
                       mean=spec.mean)


def load_images(paths, spec):
    '''Load multiple images, processed based on the given spec.'''
    return tf.pack([load_image(path, spec) for path in paths])


class ImageNet(object):
    '''Iterates over the ImageNet validation set.'''

    def __init__(self, val_path, data_path, data_spec):
        # Read in the ground truth labels for the validation set
        # The get_ilsvrc_aux.sh in Caffe's data/ilsvrc12 folder can fetch a copy of val.txt
        gt_lines = open(val_path).readlines()
        gt_pairs = [line.split() for line in gt_lines]
        # Get the full image paths
        # You will need a copy of the ImageNet validation set for this.
        self.image_paths = [osp.join(data_path, p[0]) for p in gt_pairs]
        # The corresponding ground truth labels
        self.labels = np.array([int(p[1]) for p in gt_pairs])
        # The data specifications for the model being validated (for preprocessing)
        self.data_spec = data_spec

    def batches(self, n):
        '''Yields a batch of up to n preprocessed image tensors and their ground truth labels.'''
        for i in xrange(0, len(self.image_paths), n):
            images = load_images(self.image_paths[i:i + n], self.data_spec)
            labels = self.labels[i:i + n]
            yield (images, labels)

    def __len__(self):
        '''Returns the number of instances in the validation set.'''
        return len(self.labels)
