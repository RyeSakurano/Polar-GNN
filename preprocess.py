import argparse
import numpy as np
import scipy.sparse as sp
import torch
import random
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data import *
import os

from torch.nn.functional import threshold
from scipy.sparse import linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


def preprocess_aw(path="../../data", dataset="cora", length = 7, size=100):
    edge = np.loadtxt(os.path.join(path, dataset, "edges.txt"), dtype=int).tolist()
    g = nx.Graph()
    g.add_nodes_from([i for i in range(1000)])
    g.add_edges_from(edge)
    g = dgl.from_networkx(g)
    g = dgl.to_bidirected(g)

    nodes = g.nodes()
    nodes = nodes.repeat_interleave(size)
    anon_walks = []
    rand_walks = dgl.sampling.random_walk(g, nodes, length = length - 1)[0]

    aw_dist = sp.lil_matrix((g.num_nodes(), length ** length), dtype=float)
    aw_dist_tmp = [dict() for _ in range(g.num_nodes())]
    aw_idx_map = dict()
    aw_cnt = 0

    for walk in rand_walks:
        start_idx = int(walk[0])

        anon_map = dict()
        anon_walk = np.zeros(length, dtype=int)
        node_cnt = 0
        for p in range(length):
            idx = int(walk[p])
            if idx not in anon_map:
                anon_map[idx] = node_cnt
                node_cnt += 1
            anon_walk[p] = anon_map[idx]
        anon_walks.append(anon_walk)

        anon_walk = ' '.join([str(_) for _ in anon_walk])
        if anon_walk not in aw_idx_map:
            aw_idx_map[anon_walk] = aw_cnt
            aw_cnt += 1
        if aw_idx_map[anon_walk] in aw_dist_tmp[start_idx]:
            aw_dist_tmp[start_idx][aw_idx_map[anon_walk]] += 1 / size
        else:
            aw_dist_tmp[start_idx][aw_idx_map[anon_walk]] = 1 / size

    for row in range(g.num_nodes()):
        for col in aw_dist_tmp[row]:
            aw_dist[row, col] = aw_dist_tmp[row][col]
    print(aw_cnt)
    
    pca = make_pipeline(TruncatedSVD(64), Normalizer(copy=False))
    aw_dist = pca.fit_transform(aw_dist)
    np.save(os.path.join(path, dataset, "AW_distb"), aw_dist)
    
    anon_walks = np.array(anon_walks)
    np.save(os.path.join(path, dataset, "anon_walks"), anon_walks)



if __name__ == "__main__":
    dataset_name = 'sys'
    preprocess_aw(dataset = dataset_name)
