import math

# -- Common

class KaffeError(Exception): pass

# -- Tensor ordering

# Ordering of the blobs
IDX_WEIGHTS  = 0
IDX_BIAS     = 1

# The tensors are ordered (c_o, c_i, h, w) or (n, c, h, w)
IDX_N        = 0
IDX_C        = 1
IDX_C_OUT    = 0
IDX_C_IN     = 1
IDX_H        = 2
IDX_W        = 3

# -- Shape Mappers

def make_shape(n, c, h, w):
    return (n, c, h, w)

def get_filter_output_shape(ih, iw, kh, kw, s, p, round_func):
    s = float(s)
    return map(int, map(round_func, (((ih+2*p-kw)/s)+1, ((iw+2*p-kw)/s)+1)))

def get_strided_kernel_output_shape(node, is_pooling):
    # TODO: Detect [kernel/pad/stride]_[h/w]
    input_shape = node.get_any_parent().output_shape
    k_params = node.layer.pooling_param if is_pooling else node.layer.convolution_param
    round_func = math.ceil if is_pooling else math.floor
    oh, ow = get_filter_output_shape(input_shape[IDX_H],
                                     input_shape[IDX_W],
                                     k_params.kernel_size,
                                     k_params.kernel_size,
                                     k_params.stride,
                                     k_params.pad,
                                     round_func)
    has_c_o = hasattr(k_params, 'num_output')
    c = k_params.num_output if has_c_o else input_shape[IDX_C]
    return make_shape(input_shape[IDX_N],
                      c,
                      oh,
                      ow)

def shape_not_implemented(node):
    raise NotImplementedError

def shape_input(node):
    return node.get_any_parent().output_shape

def shape_scalar(node):
    return make_shape(1, 1, 1, 1)

def shape_identity(node):
    if node.output_shape:
        return node.output_shape
    # We most likely have a data layer on our hands. The problem is,
    # Caffe infers the dimensions of the data from the source (eg: LMDB).
    # We want to avoid reading datasets here. Fail for now.
    # This can be temporarily fixed by transforming the data layer to
    # Caffe's "input" layer (as is usually used in the "deploy" version).
    # TODO: Find a better solution for this.
    raise KaffeError('Cannot determine dimensions of data layer.\n'
                     'See comments in function shape_identity for more info.')

def shape_concat(node):
    axis = node.layer.concat_param.axis
    output_shape = None
    for parent in node.parents:
        if output_shape is None:
            output_shape = list(parent.output_shape)
        else:
            output_shape[axis] += parent.output_shape[axis]
    return tuple(output_shape)

def shape_convolution(node):
    return get_strided_kernel_output_shape(node, is_pooling=False)

def shape_pool(node):
    return get_strided_kernel_output_shape(node, is_pooling=True)

def shape_inner_product(node):
    input_shape = node.get_any_parent().output_shape
    return make_shape(input_shape[IDX_N],
                      node.layer.inner_product_param.num_output,
                      1,
                      1)

# -- Layer Types

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

LAYER_TYPES = tuple(LAYER_DESCRIPTORS.keys())

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
