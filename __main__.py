import tensorflow as tf
import dgl
import networkx as nx
import matplotlib.pyplot as plt
u = tf.constant(list(range(0, 5)), dtype=tf.int64)
v = tf.constant(list(range(1, 6)), dtype=tf.int64)
star1 = dgl.graph((u, v))
star1.ndata["x"] = tf.ones((star1.num_nodes(), 3, 3))
star1.edata["x"] = tf.ones((star1.num_edges(), 3, 5))
nx.draw(star1.to_networkx(), with_labels=True)
plt.show()
print(star1)
