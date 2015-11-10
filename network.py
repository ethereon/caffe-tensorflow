import tensorflow as tf

class Network(object):
    def __init__(self, input):                
        self.vars = []
        self.batch_size = int(input.get_shape()[0])
        self.add_('input', input)
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def get_unique_name_(self, prefix):        
        id = sum(t.startswith(prefix) for t,_ in self.vars)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var):
        self.vars.append((name, var))

    def get_output(self):
        return self.vars[-1][1]

    def make_var(self, name, shape):
        return tf.get_variable(name, shape)

    def conv(self, h, w, c_i, c_o, stride=1, name=None):
        name = name or self.get_unique_name_('conv')
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[h, w, c_i, c_o])                                        
            conv = tf.nn.conv2d(self.get_output(), kernel, [stride]*4, padding='SAME')
            biases = self.make_var('biases', [c_o])
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            relu = tf.nn.relu(bias, name=scope.name)
            self.add_(name, relu)
        return self

    def pool(self, size=2, stride=2, name=None):
        name = name or self.get_unique_name_('pool')
        pool = tf.nn.max_pool(self.get_output(),
                              ksize=[1, size, size, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME',
                              name=name)
        self.add_(name, pool)
        return self

    def fc(self, num_out, relu=True, name=None):
        name = name or self.get_unique_name_('fc')
        with tf.variable_scope(name) as scope:
            input = self.get_output()
            input_shape = input.get_shape()
            if input_shape.ndims==4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [self.batch_size, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            self.add_(name, fc)
        return self

    def softmax(self, name=None):
        name = name or self.get_unique_name_('softmax')
        self.add_(name, tf.nn.softmax(self.get_output()))
        return self

