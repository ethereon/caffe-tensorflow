import os
import sys
import numpy as np
from google.protobuf import text_format

from . import caffepb
from .layers import *

class Node(object):
    def __init__(self, name, kind, layer=None):
        self.name         = name
        self.kind         = kind
        self.layer        = LayerAdapter(layer, kind) if layer else None
        self.parents      = []
        self.children     = []
        self.data         = None
        self.output_shape = None
        self.metadata     = {}

    def add_parent(self, parent_node):
        assert parent_node not in self.parents
        self.parents.append(parent_node)
        if self not in parent_node.children:
            parent_node.children.append(self)

    def add_child(self, child_node):
        assert child_node not in self.children
        self.children.append(child_node)
        if self not in child_node.parents:
            child_node.parents.append(self)

    def get_only_parent(self):
        if len(self.parents)!=1:
            raise KaffeError('Node (%s) expected to have 1 parent. Found %s.'%(self, len(self.parents)))
        return self.parents[0]

    @property
    def parameters(self):
        if self.layer is not None:
            return self.layer.parameters
        return None

    @property
    def data_shape(self):
        assert self.data
        return self.data[IDX_WEIGHTS].shape

    def __str__(self):
        return '[%s] %s'%(self.kind, self.name)

    def __repr__(self):
        return '%s (0x%x)'%(self.name, id(self))

class Graph(object):
    def __init__(self, nodes=None, name=None):
        self.nodes = nodes or []
        self.node_lut = {node.name:node for node in self.nodes}
        self.name = name

    def add_node(self, node):
        self.nodes.append(node)
        self.node_lut[node.name] = node

    def get_node(self, name):
        try:
            return self.node_lut[name]
        except KeyError:
            raise KaffeError('Layer not found: %s'%name)

    def get_input_nodes(self):
        return [node for node in self.nodes if len(node.parents)==0]

    def get_output_nodes(self):
        return [node for node in self.nodes if len(node.children)==0]

    def topologically_sorted(self):
        sorted_nodes = []
        unsorted_nodes = list(self.nodes)
        temp_marked = set()
        perm_marked = set()
        def visit(node):
            if node in temp_marked:
                raise KaffeError('Graph is not a DAG.')
            if node in perm_marked:
                return
            temp_marked.add(node)
            for child in node.children:
                visit(child)
            perm_marked.add(node)
            temp_marked.remove(node)
            sorted_nodes.insert(0, node)
        while len(unsorted_nodes):
            visit(unsorted_nodes.pop())
        return sorted_nodes

    def compute_output_shapes(self):
        sorted_nodes = self.topologically_sorted()
        for node in sorted_nodes:
            node.output_shape = NodeKind.compute_output_shape(node)

    def __contains__(self, key):
        return key in self.node_lut

    def __str__(self):
        hdr = '{:<20} {:<30} {:>20} {:>20}'.format('Type', 'Name', 'Param', 'Output')
        s = [hdr, '-'*94]
        for node in self.topologically_sorted():
            data_shape = node.data[IDX_WEIGHTS].shape if node.data else '--'
            out_shape = node.output_shape or '--'
            s.append('{:<20} {:<30} {:>20} {:>20}'.format(node.kind,
                node.name, data_shape, out_shape))
        return '\n'.join(s)

class DataInjector(object):
    def __init__(self, def_path, data_path):
        self.def_path = def_path
        self.data_path = data_path
        self.did_use_pb = False
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
        self.params = [(k, map(data, v)) for k,v in net.params.items()]

    def load_using_pb(self):
        data = caffepb.NetParameter()
        data.MergeFromString(open(self.data_path, 'rb').read())
        pair = lambda layer: (layer.name, self.transform_data(layer))
        layers = data.layers or data.layer
        self.params = [pair(layer) for layer in layers if layer.blobs]
        self.did_use_pb = True

    def transform_data(self, layer):
        transformed = []
        for idx, blob in enumerate(layer.blobs):
            if len(blob.shape.dim):
                dims = blob.shape.dim
                c_o, c_i, h, w = map(int, [1]*(4-len(dims))+list(dims))
            else:
                c_o  = blob.num
                c_i  = blob.channels
                h    = blob.height
                w    = blob.width
            data = np.array(blob.data, dtype=np.float32).reshape(c_o, c_i, h, w)
            transformed.append(data)
        return transformed

    def adjust_parameters(self, node, data):        
        if not self.did_use_pb:
            return data
        # When using the protobuf-backend, each parameter initially has four dimensions.
        # In certain cases (like FC layers), we want to eliminate the singleton dimensions.
        # This implementation takes care of the common cases. However, it does leave the
        # potential for future issues.
        # The Caffe-backend does not suffer from this problem.
        data = list(data)
        squeeze_indices = [1] # Squeeze biases.
        if node.kind==NodeKind.InnerProduct:
            squeeze_indices.append(0) # Squeeze FC.
        for idx in squeeze_indices:
            data[idx] = np.squeeze(data[idx])
        return data

    def inject(self, graph):
        for layer_name, data in self.params:
            if layer_name in graph:
                node = graph.get_node(layer_name)
                node.data = self.adjust_parameters(node, data)
            else:
                print('Ignoring parameters for non-existent layer: %s'%layer_name)

class DataReshaper(object):
    def __init__(self, mapping):
        self.mapping = mapping

    def map(self, ndim):
        try:
            return self.mapping[ndim]
        except KeyError:
            raise KaffeError('Ordering not found for %d dimensional tensor.'%ndim)

    def transpose(self, data):
        return data.transpose(self.map(data.ndim))

    def has_spatial_parent(self, node):
        try:
            parent = node.get_only_parent()
            s = parent.output_shape
            return (s[IDX_H]>1 or s[IDX_W]>1)
        except KaffeError:
            return False

    def reshape(self, graph, replace=True):
        for node in graph.nodes:
            if node.data is None:
                continue
            data = node.data[IDX_WEIGHTS]
            if (node.kind==NodeKind.InnerProduct) and self.has_spatial_parent(node):
                # The FC layer connected to the spatial layer needs to be
                # re-wired to match the new spatial ordering.
                in_shape = node.get_only_parent().output_shape
                fc_shape = data.shape
                fc_order = self.map(2)
                data = data.reshape((fc_shape[IDX_C_OUT], in_shape[IDX_C], in_shape[IDX_H], in_shape[IDX_W]))
                data = self.transpose(data)
                node.reshaped_data = data.reshape(fc_shape[fc_order[0]], fc_shape[fc_order[1]])
            else:
                node.reshaped_data = self.transpose(data)

        if replace:
            for node in graph.nodes:
                if node.data is not None:
                    node.data[IDX_WEIGHTS] = node.reshaped_data
                    del node.reshaped_data

class GraphBuilder(object):
    def __init__(self, def_path, data_path=None, phase='test'):
        self.def_path = def_path
        self.data_path = data_path
        self.phase = phase
        self.load()

    def load(self):
        self.params = caffepb.NetParameter()
        with open(self.def_path, 'rb') as def_file:
            text_format.Merge(def_file.read(), self.params)

    def filter_layers(self, layers):
        phase_map = {0:'train', 1:'test'}
        filtered_layer_names = set()
        filtered_layers = []
        for layer in layers:
            phase = self.phase
            if len(layer.include):
                phase = phase_map[layer.include[0].phase]
            if len(layer.exclude):
                phase = phase_map[1-layer.include[0].phase]
            exclude = (phase!=self.phase)
            # Dropout layers appear in a fair number of Caffe
            # test-time networks. These are just ignored. We'll
            # filter them out here.
            if (not exclude) and (phase=='test'):
                exclude = (layer.type==LayerType.Dropout)
            if not exclude:
                filtered_layers.append(layer)
                # Guard against dupes.
                assert layer.name not in filtered_layer_names
                filtered_layer_names.add(layer.name)
        return filtered_layers

    def make_node(self, layer):
        kind = NodeKind.map_raw_kind(layer.type)
        if kind is None:
            raise KaffeError('Unknown layer type encountered: %s'%kind)
        return Node(layer.name, kind, layer=layer)

    def make_input_nodes(self):
        nodes = [Node(name, NodeKind.Data) for name in self.params.input]
        if len(nodes):
            input_dim = map(int, self.params.input_dim)
            if not input_dim:
                if len(self.params.input_shape)>0:
                    input_dim = map(int, self.params.input_shape[0].dim)
                else:
                    raise KaffeError('Dimensions for input not specified.')
            for node in nodes:
                node.output_shape = tuple(input_dim)
        return nodes

    def fuse_relus(self, nodes):
        fused_nodes = []
        for node in nodes:
            if node.kind!=NodeKind.ReLU:
                continue
            parent = node.get_only_parent()
            if len(parent.children)!=1:
                # We can only fuse this ReLU if its parent's
                # value isn't used by any other node.
                continue
            # Rewrite the ReLU's children to its parent.
            for child in node.children:
                child.parents.remove(node)
                parent.add_child(child)
            # Disconnect the ReLU from the graph.
            parent.children.remove(node)
            fused_nodes.append(node)
            # Annotate the fused node.
            parent.metadata['relu'] = True
        return [node for node in nodes if node not in fused_nodes]

    def build(self, fuse_relus=True):
        layers = self.params.layers or self.params.layer
        layers = self.filter_layers(layers)
        nodes = self.make_input_nodes()
        nodes += [self.make_node(layer) for layer in layers]
        graph = Graph(nodes=nodes, name=self.params.name)
        node_outputs = {}
        for layer in layers:
            node = graph.get_node(layer.name)
            for parent_name in layer.bottom:
                assert parent_name!=layer.name
                parent_node = node_outputs.get(parent_name)
                if (parent_node is None) or (parent_node==node):
                    parent_node = graph.get_node(parent_name)
                node.add_parent(parent_node)
            for child_name in layer.top:
                if child_name==layer.name:
                    continue
                if child_name in graph:
                    # This is an "in-place operation" that overwrites an existing node.
                    # This would create a cycle in the graph. We'll undo the in-placing
                    # by substituting this node wherever the overwritten node is referenced.
                    node_outputs[child_name] = node
                else:
                    # This is an "implicit" child node: not explicitly
                    # defined in the prototxt, but as a top (output) for some layer.
                    graph.add_node(Node(child_name, NodeKind.Implicit))
                    node.add_child(graph.get_node(child_name))
        if fuse_relus:
            graph = Graph(nodes=self.fuse_relus(graph.nodes), name=graph.name)
        graph.compute_output_shapes()
        if self.data_path is not None:
            DataInjector(self.def_path, self.data_path).inject(graph)
        return graph

class NodeMapper(NodeDispatch):
    def __init__(self, graph):
        self.graph = graph

    def attach_node(self, node):
        return True

    def map(self):
        nodes = self.graph.topologically_sorted()
        # Remove input nodes - we'll handle them separately.
        input_nodes = self.graph.get_input_nodes()
        nodes = [t for t in nodes if t not in input_nodes]
        # Remove implicit nodes.
        nodes = [t for t in nodes if t.kind!=NodeKind.Implicit]
        # Decompose DAG into chains.
        chains = []
        for node in nodes:
            attach_to_chain = None
            if len(node.parents)==1:
                parent = node.get_only_parent()
                for chain in chains:
                    if chain[-1]==parent:
                        # Node is part of an existing chain.
                        attach_to_chain = chain
                        break
            if attach_to_chain is None:
                # Start a new chain for this node.
                attach_to_chain = []
                chains.append(attach_to_chain)
            attach_to_chain.append(node)
        # Map each chain.
        mapped_chains = []
        for chain in chains:
            mapped_chains.append(self.map_chain(chain))
        return self.commit(mapped_chains)

    def map_chain(self, chain):
        return [self.map_node(node) for node in chain]

    def map_node(self, node):
        map_func = self.get_handler(node.kind, 'map')
        mapped_node = map_func(node)
        assert mapped_node is not None
        if self.attach_node(node):
            mapped_node.node = node
        return mapped_node

    def commit(self, mapped_chains):
        raise NotImplementedError('Must be implemented by subclass.')
