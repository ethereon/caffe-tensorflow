import re
from .shapes import *
from collections import namedtuple

LAYER_DESCRIPTORS =  {

    # Caffe Types
    'AbsVal'                    : shape_input,
    'Accuracy'                  : shape_scalar,
    'ArgMax'                    : shape_not_implemented,
    'BNLL'                      : shape_not_implemented,
    'Concat'                    : shape_concat,
    'ContrastiveLoss'           : shape_scalar,
    'Convolution'               : shape_convolution,
    'Deconvolution'             : shape_not_implemented,
    'Data'                      : shape_identity,
    'Dropout'                   : shape_input,
    'DummyData'                 : shape_identity,
    'EuclideanLoss'             : shape_scalar,
    'Eltwise'                   : shape_input,
    'Exp'                       : shape_input,
    'Flatten'                   : shape_not_implemented,
    'HDF5Data'                  : shape_identity,
    'HDF5Output'                : shape_input,
    'HingeLoss'                 : shape_scalar,
    'Im2col'                    : shape_not_implemented,
    'ImageData'                 : shape_identity,
    'InfogainLoss'              : shape_scalar,
    'InnerProduct'              : shape_inner_product,
    'LRN'                       : shape_input,
    'MemoryData'                : shape_identity,
    'MultinomialLogisticLoss'   : shape_scalar,
    'MVN'                       : shape_not_implemented,
    'Pooling'                   : shape_pool,
    'Power'                     : shape_input,
    'ReLU'                      : shape_input,
    'Sigmoid'                   : shape_input,
    'SigmoidCrossEntropyLoss'   : shape_scalar,
    'Silence'                   : shape_not_implemented,
    'Softmax'                   : shape_input,
    'SoftmaxWithLoss'           : shape_scalar,
    'Split'                     : shape_not_implemented,
    'Slice'                     : shape_not_implemented,
    'TanH'                      : shape_input,
    'WindowData'                : shape_not_implemented,
    'Threshold'                 : shape_input,

    # Internal Types
    'Implicit'                  : shape_input
}

LAYER_TYPES = LAYER_DESCRIPTORS.keys()

def generate_layer_type_enum():
    types = {t:t for t in LAYER_TYPES}
    return type('LayerType', (), types)

LayerType = generate_layer_type_enum()

class NodeKind(LayerType):
    @staticmethod
    def map_raw_kind(kind):
        if kind in LAYER_TYPES:
            return kind
        return None

    @staticmethod
    def compute_output_shape(node):
        try:
            val = LAYER_DESCRIPTORS[node.kind](node)
            return val
        except NotImplementedError:
            raise KaffeError('Output shape computation not implemented for type: %s'%node.kind)


