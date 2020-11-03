import networkx as nx
from scipy.sparse import coo_matrix
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import utils
import model

from spektral.layers.ops import sp_matrix_to_sp_tensor
tf.debugging.experimental.enable_dump_debug_info(
    dump_root="/tmp/tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)


def build_karate_graph():
    n = 34
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
                    10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
                    25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
                    32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
                    33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
                    5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
                    24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
                    29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
                    31, 32])
    # src = np.array([1, 1])
    # dst = np.array([0, 2])
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    e = len(u)
    return coo_matrix((np.ones([e]), (u, v)), shape=(n, n))
    # return coo_matrix((1, (u, v)), shape=(34, 34))


karate_graph = build_karate_graph()

# utils.draw_scipy_kamada(karate_graph)

# print(karate_graph)


karate_x = tf.constant(
    [i for i in range(karate_graph.shape[0])], dtype=tf.int64)


karate_graph = model.GraphConv.preprocess(karate_graph).astype("f4")
# print(karate_graph)
karate_graph = sp_matrix_to_sp_tensor(karate_graph)
# labeled_nodes = tf.constant([0, 33], dtype=tf.int64)
# labels = tf.constant([0, 1], dtype=tf.int64)

karate_y = [0]*karate_graph.shape[0]
karate_y[0] = 0
karate_y[33] = 1
karate_y = tf.constant(karate_y)

mask = np.zeros([karate_graph.shape[0]])
# print(mask)
mask[0] = 1
mask[33] = 1

# mask[1:16] = 1
# mask[1:32]=0

gcn = model.GCN(n=karate_x.shape[0], embed=5)
gcn.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
gcn.fit(x=[karate_graph, karate_x], y=karate_y,
        batch_size=karate_graph.shape[0], sample_weight=mask,epochs=200)
