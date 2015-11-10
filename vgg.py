from network import Network

class VGG16(Network):
    def setup(self):
        (self.conv(3, 3,   3,  64, name='conv1_1')
             .conv(3, 3,  64,  64, name='conv1_2')
             .pool()
             .conv(3, 3,  64, 128, name='conv2_1')
             .conv(3, 3, 128, 128, name='conv2_2')
             .pool()
             .conv(3, 3, 128, 256, name='conv3_1')
             .conv(3, 3, 256, 256, name='conv3_2')
             .conv(3, 3, 256, 256, name='conv3_3')
             .pool()
             .conv(3, 3, 256, 512, name='conv4_1')
             .conv(3, 3, 512, 512, name='conv4_2')
             .conv(3, 3, 512, 512, name='conv4_3')
             .pool()
             .conv(3, 3, 512, 512, name='conv5_1')
             .conv(3, 3, 512, 512, name='conv5_2')
             .conv(3, 3, 512, 512, name='conv5_3')
             .pool()
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(1000, relu=False, name='fc8'))

