import tensorflow as tf
import numpy as np
from .core import GraphBuilder, DataReshaper, KaffeError

def to_tensorflow(def_path, data_src_path, data_dst_path, verbose=True):
    mapping = {4 : (2, 3, 1, 0), # (c_o, c_i, h, w) -> (h, w, c_i, c_o)
               2 : (1, 0)}       # (c_o, c_i) -> (c_i, c_o)
    print('Converting graph.')
    try:
        graph = GraphBuilder(def_path, data_src_path).build()
        if verbose:
            print(graph)
        DataReshaper(mapping).reshape(graph)
        params = {node.name:node.data for node in graph.nodes if node.data}
    except KaffeError as err:
        print('Error encountered: %s'%err)
        return False
    print('Saving parameters.')
    np.save(open(data_dst_path, 'wb'), params)
    print('Done.')
    return True

class TensorFlowLoader(object):
    def __init__(self, params_path):
        self.params = np.load(params_path).item()

    def load(self, sesh):
        for key in self.params:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), self.params[key]):
                    sesh.run(tf.get_variable(subkey).assign(data))
