#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
from kaffe import KaffeError
from kaffe.tensorflow import TensorFlowTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('data_path', help='Model data (.caffemodel) path')
    parser.add_argument('data_output_path', help='Converted data output path')
    parser.add_argument('code_output_path', nargs='?', help='Save generated source to this path')
    parser.add_argument('-p', '--phase', default='test', help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    try:
        transformer = TensorFlowTransformer(args.def_path, args.data_path, phase=args.phase)
        print('Converting data...')
        data = transformer.transform_data()
        print('Saving data...')
        with open(args.data_output_path, 'wb') as data_out:
            np.save(data_out, data)
        if args.code_output_path is not None:
            print('Saving source...')
            with open(args.code_output_path, 'wb') as src_out:
                src_out.write(transformer.transform_source())
        print('Done.')
    except KaffeError as err:
        print('Error encountered: %s'%err)
        exit(-1)

if __name__ == '__main__':
    main()
