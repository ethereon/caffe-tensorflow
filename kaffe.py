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

def dump_caffe(arch_path, param_path, dst_path):
    import caffe
    net = caffe.Net(arch_path, param_path, caffe.TEST)
    params = []
    def convert(blob):
        data = blob.data
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
    for key, blobs in net.params.items():
        params.append((key, map(convert, blobs)))
    print('Saving to %s'%dst_path)
    np.save(open(dst_path, 'wb'), dict(params))
    print('Done.')

def main():
    args = sys.argv[1:]
    if len(args)!=3:
        print('usage: %s path.prototxt path.caffemodel output-path'%os.path.basename(__file__))
        exit(-1)
    dump_caffe(*args)

if __name__ == '__main__':
    main()