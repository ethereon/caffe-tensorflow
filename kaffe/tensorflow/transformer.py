import tensorflow as tf
import numpy as np
from . import network
from ..base import *
from ..core import GraphBuilder, DataReshaper, NodeMapper

class TensorFlowNode(object):
    def __init__(self, op, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = list(kwargs.items())

    def format(self, arg):
        return "'%s'"%arg if isinstance(arg, basestring) else str(arg)

    def pair(self, key, value):
        return '%s=%s'%(key, self.format(value))

    def emit(self):
        args = map(self.format, self.args)        
        if self.kwargs:
            args += [self.pair(k, v) for k,v in self.kwargs]
        args.append(self.pair('name', self.node.name))
        args = ', '.join(args)
        return '%s(%s)'%(self.op, args)

def get_padding_type(kernel_params, input_shape, output_shape):
    '''Translates Caffe's numeric padding to one of ('SAME', 'VALID').
    Caffe supports arbitrary padding values, while TensorFlow only
    supports 'SAME' and 'VALID' modes. So, not all Caffe paddings
    can be translated to TensorFlow. There are some subtleties to
    how the padding edge-cases are handled. These are described here:
    https://github.com/Yangqing/caffe2/blob/master/caffe2/proto/caffe2_legacy.proto
    '''
    k_h, k_w, s_h, s_w, p_h, p_w = kernel_params
    s_o_h = np.ceil(input_shape[IDX_H]/float(s_h))
    s_o_w = np.ceil(input_shape[IDX_W]/float(s_w))
    if (output_shape[IDX_H]==s_o_h) and (output_shape[IDX_W]==s_o_w):
        return 'SAME'
    v_o_h = np.ceil((input_shape[IDX_H]-k_h+1.0)/float(s_h))
    v_o_w = np.ceil((input_shape[IDX_W]-k_w+1.0)/float(s_w))
    if (output_shape[IDX_H]==v_o_h) and (output_shape[IDX_W]==v_o_w):
        return 'VALID'
    return None

class TensorFlowMapper(NodeMapper):

    def get_kernel_params(self, node):
        kernel_params = node.layer.kernel_parameters
        input_shape = node.get_only_parent().output_shape
        padding = get_padding_type(kernel_params, input_shape, node.output_shape)
        # Only emit the padding if it's not the default value.
        padding = {'padding':padding} if padding!=network.DEFAULT_PADDING else {}
        return (kernel_params, padding)

    def relu_adapted_node(self, node, *args, **kwargs):
        # Opt-out instead of opt-in as ReLU(op) is the common case.
        if not node.metadata.get('relu', False):
            kwargs['relu']=False
        return TensorFlowNode(*args, **kwargs)

    def map_convolution(self, node):
        (c_o, c_i, h, w) = node.data_shape
        (kernel_params, kwargs) = self.get_kernel_params(node)
        group = node.parameters.group
        if group!=1:
            kwargs['group'] = group
        assert kernel_params.kernel_h==h
        assert kernel_params.kernel_w==w
        return self.relu_adapted_node(node,
                                      'conv',
                                      kernel_params.kernel_h,
                                      kernel_params.kernel_w,
                                      c_o,
                                      kernel_params.stride_h,
                                      kernel_params.stride_w,
                                      **kwargs)

    def map_relu(self, node):
        return TensorFlowNode('relu')

    def map_pooling(self, node):
        pool_type = node.parameters.pool
        if pool_type==0:
            pool_op = 'max_pool'
        elif pool_type==1:
            pool_op = 'avg_pool'
        else:
            # Stochastic pooling, for instance.
            raise KaffeError('Unsupported pooling type.')
        (kernel_params, padding) = self.get_kernel_params(node)
        return TensorFlowNode(pool_op,
                              kernel_params.kernel_h,
                              kernel_params.kernel_w,
                              kernel_params.stride_h,
                              kernel_params.stride_w,
                              **padding)

    def map_inner_product(self, node):
        #TODO: Axis
        return self.relu_adapted_node(node,
                                      'fc',
                                      node.parameters.num_output)

    def map_softmax(self, node):
        return TensorFlowNode('softmax')

    def map_lrn(self, node):
        params = node.parameters
        # The window size must be an odd value. For a window
        # size of (2*n+1), TensorFlow defines depth_radius = n.
        assert (params.local_size%2==1)
        # Caffe scales by (alpha/(2*n+1)), whereas TensorFlow
        # just scales by alpha (as does Krizhevsky's paper).
        # We'll account for that here.
        alpha = params.alpha/float(params.local_size)
        return TensorFlowNode('lrn',
                              int(params.local_size/2),
                              alpha,
                              params.beta)

    def map_concat(self, node):
        axis = (2, 3, 1, 0)[node.parameters.axis]
        return TensorFlowNode('concat', axis)

    def commit(self, chains):
        return chains

class TensorFlowEmitter(object):

    def __init__(self, tab=None):
        self.tab = tab or ' '*4
        self.prefix = ''

    def indent(self):
        self.prefix += self.tab

    def outdent(self):
        self.prefix = self.prefix[:-len(self.tab)]

    def statement(self, s):
        return self.prefix+s+'\n'

    def emit_imports(self):
        return self.statement('from kaffe.tensorflow import Network\n')

    def emit_class_def(self, name):
        return self.statement('class %s(Network):'%(name))

    def emit_setup_def(self):
        return self.statement('def setup(self):')

    def emit_parents(self, chain):
        assert len(chain)
        s = '(self.feed('
        sep = ', \n'+self.prefix+(' '*len(s))
        s += sep.join(["'%s'"%parent.name for parent in chain[0].node.parents])
        return self.statement(s+')')

    def emit_node(self, node):
        return self.statement(' '*5+'.'+node.emit())

    def emit(self, name, chains):
        s = self.emit_imports()
        s += self.emit_class_def(name)
        self.indent()
        s += self.emit_setup_def()
        self.indent()
        blocks = []
        for chain in chains:
            b = ''
            b += self.emit_parents(chain)
            for node in chain:
                b += self.emit_node(node)
            blocks.append(b[:-1]+')')
        s = s + '\n\n'.join(blocks)
        return s


class TensorFlowTransformer(object):
    def __init__(self, def_path, data_path, verbose=True):
        self.data_reshaped = False
        self.verbose = verbose
        self.load(def_path, data_path)
        self.source = None

    def load(self, def_path, data_path):
        self.graph = GraphBuilder(def_path, data_path).build()
        for node in self.graph.nodes:
            # Slashes are used for scoping in TensorFlow. Replace slashes
            # in node names with underscores.
            # (Caffe's GoogLeNet implementation uses slashes)
            node.name = node.name.replace('/', '_')
        if self.verbose:
            print(self.graph)

    def transform_data(self):    
        # Cache the graph source before mutating it.
        self.transform_source()        
        mapping = {4 : (2, 3, 1, 0), # (c_o, c_i, h, w) -> (h, w, c_i, c_o)
                   2 : (1, 0)}       # (c_o, c_i) -> (c_i, c_o)
        DataReshaper(mapping).reshape(self.graph)
        return {node.name:node.data for node in self.graph.nodes if node.data}

    def transform_source(self):
        if self.source is None:
            mapper = TensorFlowMapper(self.graph)
            chains = mapper.map()
            emitter = TensorFlowEmitter()
            self.source = emitter.emit(self.graph.name, chains)
        return self.source
