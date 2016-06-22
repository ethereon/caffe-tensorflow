'''Utility functions and classes for handling image datasets.'''

import os.path as osp
import numpy as np
import tensorflow as tf


def process_image(img, scale, isotropic, crop, mean):
    '''Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, a central crop of this size is taken.
    mean  : Subtracted from the image
    '''
    # Rescale
    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.pack([scale, scale])
    img = tf.image.resize_images(img, new_shape[0], new_shape[1])
    # Center crop
    # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
    # See: https://github.com/tensorflow/tensorflow/issues/521
    offset = (new_shape - crop) / 2
    img = tf.slice(img, begin=tf.pack([offset[0], offset[1], 0]), size=tf.pack([crop, crop, -1]))
    # Mean subtraction
    return tf.to_float(img) - mean


class ImageProducer(object):
    '''
    Loads and processes batches of images in parallel.
    '''

    def __init__(self, image_paths, data_spec, num_concurrent=4, batch_size=None, labels=None):
        # The data specifications describe how to process the image
        self.data_spec = data_spec
        # A list of full image paths
        self.image_paths = image_paths
        # An optional list of labels corresponding to each image path
        self.labels = labels
        # A boolean flag per image indicating whether its a JPEG or PNG
        self.extension_mask = self.create_extension_mask(self.image_paths)
        # Create the loading and processing operations
        self.setup(batch_size=batch_size, num_concurrent=num_concurrent)

    def setup(self, batch_size, num_concurrent):
        # Validate the batch size
        num_images = len(self.image_paths)
        batch_size = min(num_images, batch_size or self.data_spec.batch_size)
        if num_images % batch_size != 0:
            raise ValueError(
                'The total number of images ({}) must be divisible by the batch size ({}).'.format(
                    num_images, batch_size))
        self.num_batches = num_images / batch_size

        # Create a queue that will contain image paths (and their indices and extension indicator)
        self.path_queue = tf.FIFOQueue(capacity=num_images,
                                       dtypes=[tf.int32, tf.bool, tf.string],
                                       name='path_queue')

        # Enqueue all image paths, along with their indices
        indices = tf.range(num_images)
        self.enqueue_paths_op = self.path_queue.enqueue_many([indices, self.extension_mask,
                                                              self.image_paths])
        # Close the path queue (no more additions)
        self.close_path_queue_op = self.path_queue.close()

        # Create an operation that dequeues a single path and returns a processed image
        (idx, processed_image) = self.process()

        # Create a queue that will contain the processed images (and their indices)
        image_shape = (self.data_spec.crop_size, self.data_spec.crop_size, self.data_spec.channels)
        processed_queue = tf.FIFOQueue(capacity=int(np.ceil(num_images / float(num_concurrent))),
                                       dtypes=[tf.int32, tf.float32],
                                       shapes=[(), image_shape],
                                       name='processed_queue')

        # Enqueue the processed image and path
        enqueue_processed_op = processed_queue.enqueue([idx, processed_image])

        # Create a dequeue op that fetches a batch of processed images off the queue
        self.dequeue_op = processed_queue.dequeue_many(batch_size)

        # Create a queue runner to perform the processing operations in parallel
        num_concurrent = min(num_concurrent, num_images)
        self.queue_runner = tf.train.QueueRunner(processed_queue,
                                                 [enqueue_processed_op] * num_concurrent)

    def start(self, session, coordinator, num_concurrent=4):
        '''Start the processing worker threads.'''
        # Queue all paths
        session.run(self.enqueue_paths_op)
        # Close the path queue
        session.run(self.close_path_queue_op)
        # Start the queue runner and return the created threads
        return self.queue_runner.create_threads(session, coord=coordinator, start=True)

    def get(self, session):
        '''
        Get a single batch of images along with their indices. If a set of labels were provided,
        the corresponding labels are returned instead of the indices.
        '''
        (indices, images) = session.run(self.dequeue_op)
        if self.labels is not None:
            labels = [self.labels[idx] for idx in indices]
            return (labels, images)
        return (indices, images)

    def batches(self, session):
        '''Yield a batch until no more images are left.'''
        for _ in xrange(self.num_batches):
            yield self.get(session=session)

    def load_image(self, image_path, is_jpeg):
        # Read the file
        file_data = tf.read_file(image_path)
        # Decode the image data
        img = tf.cond(
            is_jpeg,
            lambda: tf.image.decode_jpeg(file_data, channels=self.data_spec.channels),
            lambda: tf.image.decode_png(file_data, channels=self.data_spec.channels))
        if self.data_spec.expects_bgr:
            # Convert from RGB channel ordering to BGR
            # This matches, for instance, how OpenCV orders the channels.
            img = tf.reverse(img, [False, False, True])
        return img

    def process(self):
        # Dequeue a single image path
        idx, is_jpeg, image_path = self.path_queue.dequeue()
        # Load the image
        img = self.load_image(image_path, is_jpeg)
        # Process the image
        processed_img = process_image(img=img,
                                      scale=self.data_spec.scale_size,
                                      isotropic=self.data_spec.isotropic,
                                      crop=self.data_spec.crop_size,
                                      mean=self.data_spec.mean)
        # Return the processed image, along with its index
        return (idx, processed_img)

    @staticmethod
    def create_extension_mask(paths):

        def is_jpeg(path):
            extension = osp.splitext(path)[-1].lower()
            if extension in ('.jpg', '.jpeg'):
                return True
            if extension != '.png':
                raise ValueError('Unsupported image format: {}'.format(extension))
            return False

        return [is_jpeg(p) for p in paths]

    def __len__(self):
        return len(self.image_paths)


class ImageNetProducer(ImageProducer):

    def __init__(self, val_path, data_path, data_spec):
        # Read in the ground truth labels for the validation set
        # The get_ilsvrc_aux.sh in Caffe's data/ilsvrc12 folder can fetch a copy of val.txt
        gt_lines = open(val_path).readlines()
        gt_pairs = [line.split() for line in gt_lines]
        # Get the full image paths
        # You will need a copy of the ImageNet validation set for this.
        image_paths = [osp.join(data_path, p[0]) for p in gt_pairs]
        # The corresponding ground truth labels
        labels = np.array([int(p[1]) for p in gt_pairs])
        # Initialize base
        super(ImageNetProducer, self).__init__(image_paths=image_paths,
                                               data_spec=data_spec,
                                               labels=labels)
