from googlenet import GoogleNet
from vgg import VGG16
from alexnet import AlexNet
from caffenet import CaffeNet

class NetConfig(object):
    def __init__(self, net, batch_size, scale_size, crop_size, isotropic, channels=3):
        self.net_class  = net
        self.batch_size = batch_size
        self.scale_size = scale_size
        self.crop_size  = crop_size
        self.isotropic  = isotropic
        self.channels   = channels

    def __str__(self):
        return self.net_class.__name__

MODELS = [

    NetConfig(VGG16,
              batch_size=25,
              scale_size=256,
              crop_size=224,
              isotropic=True),

    NetConfig(GoogleNet,
              batch_size=200,
              scale_size=256,
              crop_size=224,
              isotropic=False),

    NetConfig(AlexNet,
              batch_size=500,
              scale_size=256,
              crop_size=227,
              isotropic=False),

    NetConfig(CaffeNet,
              batch_size=500,
              scale_size=256,
              crop_size=227,
              isotropic=False)
]
