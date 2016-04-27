# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mynet import LeNet as MyNet

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 32

def gen_data(source):
    while True:
        indices = range(len(source.images))
        random.shuffle(indices)
        for i in indices:
            image = np.reshape(source.images[i], (28, 28, 1))
            label = source.labels[i]
            yield image, label

def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        label_batch = []
        for _ in range(batch_size):
            image, label = next(data_gen)
            image_batch.append(image)
            label_batch.append(label)
        yield np.array(image_batch), np.array(label_batch)


images = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])
labels = tf.placeholder(tf.float32, [batch_size, 10])
net = MyNet({'data': images})

ip2 = net.layers['ip2']
pred = tf.nn.softmax(ip2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(ip2, labels), 0)
opt = tf.train.RMSPropOptimizer(0.001)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    # Load the data
    sess.run(tf.initialize_all_variables())
    net.load('mynet.npy', sess)

    data_gen = gen_data_batch(mnist.train)
    for i in range(1000):
        np_images, np_labels = next(data_gen)
        feed = {images: np_images, labels: np_labels}

        np_loss, np_pred, _ = sess.run([loss, pred, train_op], feed_dict=feed)
        if i % 10 == 0:
            print('Iteration: ', i, np_loss)
