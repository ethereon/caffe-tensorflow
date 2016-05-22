import numpy as np

from .caffe import get_caffe_resolver, has_pycaffe
from .errors import KaffeError, print_stderr
from .layers import NodeKind

class DataInjector(object):

    def __init__(self, def_path, data_path):
        self.def_path = def_path
        self.data_path = data_path
        self.did_use_pb = False
        self.params = None
        self.load()

    def load(self):
        if has_pycaffe():
            self.load_using_caffe()
        else:
            self.load_using_pb()

    def load_using_caffe(self):
        caffe = get_caffe_resolver().caffe
        net = caffe.Net(self.def_path, self.data_path, caffe.TEST)
        data = lambda blob: blob.data
        self.params = [(k, map(data, v)) for k, v in net.params.items()]

    def load_using_pb(self):
        data = get_caffe_resolver().NetParameter()
        data.MergeFromString(open(self.data_path, 'rb').read())
        pair = lambda layer: (layer.name, self.transform_data(layer))
        layers = data.layers or data.layer
        self.params = [pair(layer) for layer in layers if layer.blobs]
        self.did_use_pb = True

    def transform_data(self, layer):
        transformed = []
        for blob in layer.blobs:
            if len(blob.shape.dim):
                dims = blob.shape.dim
                c_o, c_i, h, w = map(int, [1] * (4 - len(dims)) + list(dims))
            else:
                c_o = blob.num
                c_i = blob.channels
                h = blob.height
                w = blob.width
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
        squeeze_indices = [1]  # Squeeze biases.
        if node.kind == NodeKind.InnerProduct:
            squeeze_indices.append(0)  # Squeeze FC.
        for idx in squeeze_indices:
            data[idx] = np.squeeze(data[idx])
        return data

    def inject(self, graph):
        for layer_name, data in self.params:
            if layer_name in graph:
                node = graph.get_node(layer_name)
                node.data = self.adjust_parameters(node, data)
            else:
                print_stderr('Ignoring parameters for non-existent layer: %s' % layer_name)


class DataReshaper(object):

    def __init__(self, mapping):
        self.mapping = mapping

    def map(self, ndim):
        try:
            return self.mapping[ndim]
        except KeyError:
            raise KaffeError('Ordering not found for %d dimensional tensor.' % ndim)

    def transpose(self, data):
        return data.transpose(self.map(data.ndim))

    def has_spatial_parent(self, node):
        try:
            parent = node.get_only_parent()
            s = parent.output_shape
            return s.height > 1 or s.width > 1
        except KaffeError:
            return False

    def reshape(self, graph, replace=True):
        for node in graph.nodes:
            if node.data is None:
                continue
            # Get the weights
            data = node.data[0]
            if (node.kind == NodeKind.InnerProduct) and self.has_spatial_parent(node):
                # The FC layer connected to the spatial layer needs to be
                # re-wired to match the new spatial ordering.
                in_shape = node.get_only_parent().output_shape
                fc_shape = ParamShape(data.shape)
                fc_order = self.map(2)
                data = data.reshape((fc_shape.output_channels, in_shape.channels, in_shape.height,
                                     in_shape.width))
                data = self.transpose(data)
                node.reshaped_data = data.reshape(fc_shape[fc_order[0]], fc_shape[fc_order[1]])
            else:
                node.reshaped_data = self.transpose(data)

        if replace:
            for node in graph.nodes:
                if node.data is not None:
                    # Set the weights
                    node.data[0] = node.reshaped_data
                    del node.reshaped_data
