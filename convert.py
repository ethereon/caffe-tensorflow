#!/usr/bin/env python

import os
import sys
from kaffe.transformers import to_tensorflow

def main():
    args = sys.argv[1:]
    if len(args)!=3:
        print('usage: %s path.prototxt path.caffemodel output-path'%os.path.basename(__file__))
        exit(-1)
    success = to_tensorflow(*args)
    exit(0 if success else -1)

if __name__ == '__main__':
    main()
