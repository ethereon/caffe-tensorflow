from kaffe.tensorflow import Network

class GoogleNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, name='conv1_7x7_s2')
             .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
             .lrn(2, 2e-05, 0.75, name='pool1_norm1')
             .conv(1, 1, 64, 1, 1, name='conv2_3x3_reduce')
             .conv(3, 3, 192, 1, 1, name='conv2_3x3')
             .lrn(2, 2e-05, 0.75, name='conv2_norm2')
             .max_pool(3, 3, 2, 2, name='pool2_3x3_s2')
             .conv(1, 1, 64, 1, 1, name='inception_3a_1x1'))

        (self.feed('pool2_3x3_s2')
             .conv(1, 1, 96, 1, 1, name='inception_3a_3x3_reduce')
             .conv(3, 3, 128, 1, 1, name='inception_3a_3x3'))

        (self.feed('pool2_3x3_s2')
             .conv(1, 1, 16, 1, 1, name='inception_3a_5x5_reduce')
             .conv(5, 5, 32, 1, 1, name='inception_3a_5x5'))

        (self.feed('pool2_3x3_s2')
             .max_pool(3, 3, 1, 1, name='inception_3a_pool')
             .conv(1, 1, 32, 1, 1, name='inception_3a_pool_proj'))

        (self.feed('inception_3a_1x1',
                   'inception_3a_3x3',
                   'inception_3a_5x5',
                   'inception_3a_pool_proj')
             .concat(3, name='inception_3a_output')
             .conv(1, 1, 128, 1, 1, name='inception_3b_1x1'))

        (self.feed('inception_3a_output')
             .conv(1, 1, 128, 1, 1, name='inception_3b_3x3_reduce')
             .conv(3, 3, 192, 1, 1, name='inception_3b_3x3'))

        (self.feed('inception_3a_output')
             .conv(1, 1, 32, 1, 1, name='inception_3b_5x5_reduce')
             .conv(5, 5, 96, 1, 1, name='inception_3b_5x5'))

        (self.feed('inception_3a_output')
             .max_pool(3, 3, 1, 1, name='inception_3b_pool')
             .conv(1, 1, 64, 1, 1, name='inception_3b_pool_proj'))

        (self.feed('inception_3b_1x1',
                   'inception_3b_3x3',
                   'inception_3b_5x5',
                   'inception_3b_pool_proj')
             .concat(3, name='inception_3b_output')
             .max_pool(3, 3, 2, 2, name='pool3_3x3_s2')
             .conv(1, 1, 192, 1, 1, name='inception_4a_1x1'))

        (self.feed('pool3_3x3_s2')
             .conv(1, 1, 96, 1, 1, name='inception_4a_3x3_reduce')
             .conv(3, 3, 208, 1, 1, name='inception_4a_3x3'))

        (self.feed('pool3_3x3_s2')
             .conv(1, 1, 16, 1, 1, name='inception_4a_5x5_reduce')
             .conv(5, 5, 48, 1, 1, name='inception_4a_5x5'))

        (self.feed('pool3_3x3_s2')
             .max_pool(3, 3, 1, 1, name='inception_4a_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4a_pool_proj'))

        (self.feed('inception_4a_1x1',
                   'inception_4a_3x3',
                   'inception_4a_5x5',
                   'inception_4a_pool_proj')
             .concat(3, name='inception_4a_output')
             .conv(1, 1, 160, 1, 1, name='inception_4b_1x1'))

        (self.feed('inception_4a_output')
             .conv(1, 1, 112, 1, 1, name='inception_4b_3x3_reduce')
             .conv(3, 3, 224, 1, 1, name='inception_4b_3x3'))

        (self.feed('inception_4a_output')
             .conv(1, 1, 24, 1, 1, name='inception_4b_5x5_reduce')
             .conv(5, 5, 64, 1, 1, name='inception_4b_5x5'))

        (self.feed('inception_4a_output')
             .max_pool(3, 3, 1, 1, name='inception_4b_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4b_pool_proj'))

        (self.feed('inception_4b_1x1',
                   'inception_4b_3x3',
                   'inception_4b_5x5',
                   'inception_4b_pool_proj')
             .concat(3, name='inception_4b_output')
             .conv(1, 1, 128, 1, 1, name='inception_4c_1x1'))

        (self.feed('inception_4b_output')
             .conv(1, 1, 128, 1, 1, name='inception_4c_3x3_reduce')
             .conv(3, 3, 256, 1, 1, name='inception_4c_3x3'))

        (self.feed('inception_4b_output')
             .conv(1, 1, 24, 1, 1, name='inception_4c_5x5_reduce')
             .conv(5, 5, 64, 1, 1, name='inception_4c_5x5'))

        (self.feed('inception_4b_output')
             .max_pool(3, 3, 1, 1, name='inception_4c_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4c_pool_proj'))

        (self.feed('inception_4c_1x1',
                   'inception_4c_3x3',
                   'inception_4c_5x5',
                   'inception_4c_pool_proj')
             .concat(3, name='inception_4c_output')
             .conv(1, 1, 112, 1, 1, name='inception_4d_1x1'))

        (self.feed('inception_4c_output')
             .conv(1, 1, 144, 1, 1, name='inception_4d_3x3_reduce')
             .conv(3, 3, 288, 1, 1, name='inception_4d_3x3'))

        (self.feed('inception_4c_output')
             .conv(1, 1, 32, 1, 1, name='inception_4d_5x5_reduce')
             .conv(5, 5, 64, 1, 1, name='inception_4d_5x5'))

        (self.feed('inception_4c_output')
             .max_pool(3, 3, 1, 1, name='inception_4d_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4d_pool_proj'))

        (self.feed('inception_4d_1x1',
                   'inception_4d_3x3',
                   'inception_4d_5x5',
                   'inception_4d_pool_proj')
             .concat(3, name='inception_4d_output')
             .conv(1, 1, 256, 1, 1, name='inception_4e_1x1'))

        (self.feed('inception_4d_output')
             .conv(1, 1, 160, 1, 1, name='inception_4e_3x3_reduce')
             .conv(3, 3, 320, 1, 1, name='inception_4e_3x3'))

        (self.feed('inception_4d_output')
             .conv(1, 1, 32, 1, 1, name='inception_4e_5x5_reduce')
             .conv(5, 5, 128, 1, 1, name='inception_4e_5x5'))

        (self.feed('inception_4d_output')
             .max_pool(3, 3, 1, 1, name='inception_4e_pool')
             .conv(1, 1, 128, 1, 1, name='inception_4e_pool_proj'))

        (self.feed('inception_4e_1x1',
                   'inception_4e_3x3',
                   'inception_4e_5x5',
                   'inception_4e_pool_proj')
             .concat(3, name='inception_4e_output')
             .max_pool(3, 3, 2, 2, name='pool4_3x3_s2')
             .conv(1, 1, 256, 1, 1, name='inception_5a_1x1'))

        (self.feed('pool4_3x3_s2')
             .conv(1, 1, 160, 1, 1, name='inception_5a_3x3_reduce')
             .conv(3, 3, 320, 1, 1, name='inception_5a_3x3'))

        (self.feed('pool4_3x3_s2')
             .conv(1, 1, 32, 1, 1, name='inception_5a_5x5_reduce')
             .conv(5, 5, 128, 1, 1, name='inception_5a_5x5'))

        (self.feed('pool4_3x3_s2')
             .max_pool(3, 3, 1, 1, name='inception_5a_pool')
             .conv(1, 1, 128, 1, 1, name='inception_5a_pool_proj'))

        (self.feed('inception_5a_1x1',
                   'inception_5a_3x3',
                   'inception_5a_5x5',
                   'inception_5a_pool_proj')
             .concat(3, name='inception_5a_output')
             .conv(1, 1, 384, 1, 1, name='inception_5b_1x1'))

        (self.feed('inception_5a_output')
             .conv(1, 1, 192, 1, 1, name='inception_5b_3x3_reduce')
             .conv(3, 3, 384, 1, 1, name='inception_5b_3x3'))

        (self.feed('inception_5a_output')
             .conv(1, 1, 48, 1, 1, name='inception_5b_5x5_reduce')
             .conv(5, 5, 128, 1, 1, name='inception_5b_5x5'))

        (self.feed('inception_5a_output')
             .max_pool(3, 3, 1, 1, name='inception_5b_pool')
             .conv(1, 1, 128, 1, 1, name='inception_5b_pool_proj'))

        (self.feed('inception_5b_1x1',
                   'inception_5b_3x3',
                   'inception_5b_5x5',
                   'inception_5b_pool_proj')
             .concat(3, name='inception_5b_output')
             .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5_7x7_s1')
             .fc(1000, relu=False, name='loss3_classifier')
             .softmax(name='prob'))
