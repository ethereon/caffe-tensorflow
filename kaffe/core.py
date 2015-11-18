import os
import sys
import numpy as np
from google.protobuf import text_format

from . import caffepb
from .types import *

class Node(object):
    def __init__(self, name, kind, layer=None):
        self.name = name
        self.kind = kind
        self.layer = layer
        self.parents = set()
        self.children = set()
        self.data = None
        self.output_shape = None

    def add_parent(self, parent_node):
        self.parents.add(parent_node)
        parent_node.children.add(self)

    def add_child(self, child_node):
        self.children.add(child_node)
        child_node.parents.add(self)

    def get_any_parent(self):
        if len(self.parents):
            return next(iter(self.parents))
        return None

    def __str__(self):
        data_shape = self.data[IDX_WEIGHTS].shape if self.data else '--'
        out_shape = self.output_shape or '--'
        return '{:<20} {:<30} {:>20} {:>20}'.format(self.kind, self.name, data_shape, out_shape)

class Graph(object):
    def __init__(self, nodes=None):
        self.nodes = nodes or []
        self.node_lut = {node.name:node for node in self.nodes}

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
        sorted_nodes = self.topologically_sorted()
        s += ['%s'%node for node in sorted_nodes]
        return '\n'.join(s)

class DataInjector(object):
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
        self.params = [(k, map(data, v)) for k,v in net.params.items()]

    def load_using_pb(self):
        data = caffepb.NetParameter()
        data.MergeFromString(open(self.data_path, 'rb').read())
        pair = lambda layer: (layer.name, self.transform_data(layer))
        layers = data.layers or data.layer
        self.params = [pair(layer) for layer in layers if layer.blobs]

    def transform_data(self, layer):
        transformed = []
        for idx, blob in enumerate(layer.blobs):
            c_o  = blob.num
            c_i  = blob.channels
            h    = blob.height
            w    = blob.width
            data = np.squeeze(np.array(blob.data, dtype=np.float32).reshape(c_o, c_i, h, w))
            transformed.append(data)
        return transformed

    def inject(self, graph):
        for layer_name, data in self.params:
            graph.get_node(layer_name).data = data

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
        parent = node.get_any_parent()
        if parent is None:
            return False
        s = parent.output_shape
        return (s[IDX_H]>1 or s[IDX_W]>1)

    def reshape(self, graph, replace=True):
        for node in graph.nodes:
            if node.data is None:
                continue
            data = node.data[IDX_WEIGHTS]
            if (node.kind==NodeKind.InnerProduct) and self.has_spatial_parent(node):
                # The FC layer connected to the spatial layer needs to be
                # re-wired to match the new spatial ordering.
                in_shape = node.get_any_parent().output_shape
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
    def __init__(self, def_path, data_path=None):
        self.def_path = def_path
        self.data_path = data_path
        self.load()

    def load(self):
        self.params = caffepb.NetParameter()
        with open(self.def_path, 'rb') as def_file:
            text_format.Merge(def_file.read(), self.params)

    def remove_duplicates(self, nodes):
        # Duplicate nodes can exist as a result of Caffe's
        # "phase" mechanism. Currently, we assume that duplicate
        # nodes are structurally similar and arbitrarily select one.
        return ({t.name:t for t in nodes}).values()

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

    def build(self):
        layers = self.params.layers or self.params.layer
        nodes = self.make_input_nodes()
        nodes += [self.make_node(layer) for layer in layers]
        nodes = self.remove_duplicates(nodes)
        graph = Graph(nodes=nodes)
        inplace_replacements = {}
        for layer in layers:
            node = graph.get_node(layer.name)
            unique_parent = layer.bottom[0] if len(layer.bottom) else None
            for child in layer.top:
                if child==unique_parent:
                    # This is an inplace node. Substitute the parent
                    # for this node in all future connections.
                    inplace_replacements[unique_parent] = node
                    # Skip this edge (which would produce a cycle).
                    continue
                if child!=layer.name:
                    if child not in graph:
                        # This is an "implicit" child node: not explicitly
                        # defined in the prototxt, but as a top for some layer.
                        graph.add_node(Node(child, NodeKind.Implicit))
                    node.add_child(graph.get_node(child))
            for parent in layer.bottom:
                assert parent!=layer.name
                parent_node = inplace_replacements.get(parent)
                if (parent_node is None) or (parent_node==node):
                    parent_node = graph.get_node(parent)
                node.add_parent(parent_node)
        graph.compute_output_shapes()
        if self.data_path is not None:
            DataInjector(self.def_path, self.data_path).inject(graph)
        return graph
