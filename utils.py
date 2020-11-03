import networkx as nx
import matplotlib.pyplot as plt


def draw_scipy_kamada(g):
    nx_g = nx.from_scipy_sparse_matrix(g)
    nx.draw_kamada_kawai(nx_g)
    plt.show()
