#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
from kaffe import KaffeError
from kaffe.tensorflow import TensorFlowTransformer

def convert(def_path,
            caffemodel=None,
            data_output_path='mynet.npy',
            code_output_path='mynet.py',
            phase='test'):
    try:
        transformer = TensorFlowTransformer(def_path, caffemodel, phase=phase)
        print('Converting data...')
        if caffemodel is not None:
            data = transformer.transform_data()
            print('Saving data...')
            with open(data_output_path, 'wb') as data_out:
                np.save(data_out, data)
        print('Saving source...')
        with open(code_output_path, 'wb') as src_out:
            src_out.write(transformer.transform_source())
        print('Done.')
    except KaffeError as err:
        print('Error encountered: %s'%err)
        exit(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', default=None, help='Model data (.caffemodel) path')
    parser.add_argument('--data_output_path', default='mynet.npy', help='Converted data output path')
    parser.add_argument('--code_output_path', default='mynet.py',
                        help='Save generated source to this path')
    parser.add_argument('-p', '--phase', default='test',
                        help='The phase to convert: test (default) or train')
    args = parser.parse_args()

    convert(args.def_path, args.caffemodel, args.data_output_path, args.code_output_path, args.phase)

if __name__ == '__main__':
    main()
