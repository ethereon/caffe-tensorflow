from kaffe.tensorflow import Network

class CaffeNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(1000, relu=False, name='fc8')
             .softmax(name='prob'))
