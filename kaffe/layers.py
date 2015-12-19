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
    'MemoryData'                : shape_mem_data,
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

class NodeDispatchError(KaffeError): pass

class NodeDispatch(object):
    def get_handler_name(self, node_kind):
        if len(node_kind)<=4:
            # A catch-all for things like ReLU and tanh
            return node_kind.lower()
        # Convert from CamelCase to under_scored
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', node_kind)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def get_handler(self, node_kind, prefix):
        name = self.get_handler_name(node_kind)
        name = '_'.join((prefix, name))
        if hasattr(self, name):
            return getattr(self, name)
        raise NodeDispatchError('No handler found for node kind: %s (expected: %s)'%(node_kind, name))

class LayerAdapter(NodeDispatch):
    def __init__(self, layer, kind):
        self.layer = layer
        self.kind = kind

    def parameters_convolution(self):
        return self.layer.convolution_param

    def parameters_pooling(self):
        return self.layer.pooling_param

    def parameters_inner_product(self):
        return self.layer.inner_product_param

    def parameters_concat(self):
        return self.layer.concat_param

    def parameters_lrn(self):
        return self.layer.lrn_param

    def parameters_memory_data(self):
        return self.layer.memory_data_param

    @property
    def parameters(self):
        handler = self.get_handler(self.kind, 'parameters')
        return handler()

    @property
    def kernel_parameters(self):
        assert self.kind in (NodeKind.Convolution, NodeKind.Pooling)
        params = self.parameters
        k_h = params.kernel_h or params.kernel_size
        k_w = params.kernel_w or params.kernel_size
        s_h = params.stride_h or params.stride
        s_w = params.stride_w or params.stride
        p_h = params.pad_h or params.pad
        p_w = params.pad_h or params.pad
        return KernelParameters(k_h, k_w, s_h, s_w, p_h, p_w)

KernelParameters = namedtuple('KernelParameters',
                              ['kernel_h',
                              'kernel_w',
                              'stride_h',
                              'stride_w',
                              'pad_h',
                              'pad_w'])
