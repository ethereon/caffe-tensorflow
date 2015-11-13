#!/usr/bin/env python

import os
import sys
import numpy as np
import tensorflow as tf

class CaffeLoader(object):
    def __init__(self, params_path):
        self.params = np.load(params_path).item()

    def load(self, sesh):
        for key in self.params:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), self.params[key]):
                    sesh.run(tf.get_variable(subkey).assign(data))

class CaffeDataReader(object):
    def __init__(self, def_path, data_path):
        self.def_path = def_path
        self.data_path = data_path
        self.load()

    def load(self):
        try:            
            self.load_using_caffe()
        except ImportError:
            print('WARNING: PyCaffe not found!')
            print('Falling back to protocol buffer implementation.')
            print('This may take a couple of minutes.')            
            self.load_using_pb()

    def load_using_caffe(self):
        import caffe
        net = caffe.Net(self.def_path, self.data_path, caffe.TEST)
        data = lambda blob: blob.data
        self.parameters = [(k, map(data, v)) for k,v in net.params.items()]        

    def load_using_pb(self):
        import caffepb
        data = caffepb.NetParameter()
        data.MergeFromString(open(self.data_path, 'rb').read())
        pair = lambda layer: (layer.name, self.transform_data(layer))
        self.parameters = [pair(layer) for layer in data.layers if layer.blobs]

    def transform_data(self, layer):
        transformed = []
        for idx, blob in enumerate(layer.blobs):
            c_o  = blob.num
            c_i  = blob.channels
            h    = blob.height
            w    = blob.width
            data = np.squeeze(np.array(blob.data).reshape(c_o, c_i, h, w))
            transformed.append(data)
        return tuple(transformed)

    def dump(self, dst_path):
        params = []
        def convert(data):
            if data.ndim==4:
                # (c_o, c_i, h, w) -> (h, w, c_i, c_o)
                data = data.transpose((2, 3, 1, 0))
            elif data.ndim==2:
                # (c_o, c_i) -> (c_i, c_o)
                data = data.transpose((1, 0))
            # Kludge Alert: The FC layer wired up to the pool layer
            # needs to be re-ordered.
            # TODO: Figure out a more elegant way to do this.        
            if params and params[-1][1][0].ndim==4 and data.ndim==2:
                prev_c_o = params[-1][1][0].shape[-1]
                cur_c_i, cur_c_o = data.shape
                dim = np.sqrt(cur_c_i/prev_c_o)
                data = data.reshape((prev_c_o, dim, dim, cur_c_o))
                data = data.transpose((1, 2, 0, 3))
                data = data.reshape((prev_c_o*dim*dim, cur_c_o))
            return data
        for key, data_pair in self.parameters:
            params.append((key, map(convert, data_pair)))
        print('Saving to %s'%dst_path)
        np.save(open(dst_path, 'wb'), dict(params))
        print('Done.')

def main():
    args = sys.argv[1:]
    if len(args)!=3:
        print('usage: %s path.prototxt path.caffemodel output-path'%os.path.basename(__file__))
        exit(-1)
    def_path, data_path, dst_path = args
    CaffeDataReader(def_path, data_path).dump(dst_path)

if __name__ == '__main__':
    main()
