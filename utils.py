import argparse
import numpy as np
from numpy.lib.type_check import real
import scipy.sparse as sp
import torch
import random
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data import *
import os
from sklearn.metrics import f1_score, confusion_matrix


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(path="../../data", dataset="cora", train_ratio=0, pos_method = "anchor_spd", struct_method = "AW_distb"):
    edge = np.loadtxt(os.path.join(path, dataset, "edges.txt"), dtype=int).tolist()
    if dataset in ['pubmed', 'twitch']:
        feat = np.load(os.path.join(path, dataset, "features.npy"), allow_pickle=True)[()].toarray().astype(np.float64)
    else:
        feat = np.load(os.path.join(path, dataset, "features.npy")).astype(np.float64)
    
    if "sys" in dataset:
        labels = np.load(os.path.join(path, dataset, "labels.npy"))
    else:
        y_dict = dict()
        for line in open(os.path.join(path, dataset, "labels.txt")):
            line = line.strip().strip('\n').split(' ')
            y_dict[int(line[0])] = int(line[1])
        labels = list()
        num_of_nodes = max(y_dict.keys()) + 1
        for i in range(num_of_nodes):
            labels.append(y_dict[i])
        labels = np.array(labels)
        labels = encode_onehot(labels)
        labels = np.where(labels)[1]

    if dataset in ['chameleon-5', 'squirrel-5', 'film']:
        idx_train = np.loadtxt(os.path.join(path, dataset, "train_{}_idx.txt".format(train_ratio)), dtype=int)
        idx_val = np.loadtxt(os.path.join(path, dataset, "val_{}_idx.txt".format(train_ratio)), dtype=int)
        idx_test = np.loadtxt(os.path.join(path, dataset, "test_{}_idx.txt".format(train_ratio)), dtype=int)
    else:
        idx_train = np.loadtxt(os.path.join(path, dataset, "train_fixed_idx.txt"), dtype=int)
        idx_val = np.loadtxt(os.path.join(path, dataset, "val_idx.txt"), dtype=int)
        idx_test = np.loadtxt(os.path.join(path, dataset, "test_idx.txt"), dtype=int)
    nclass = len(set(labels.tolist()))
    print(dataset, nclass)

    g = nx.Graph()
    g.add_nodes_from([i for i in range(feat.shape[0])])
    g.add_edges_from(edge)
    g = dgl.from_networkx(g)
    g = dgl.to_bidirected(g)

    
    structure = np.load(os.path.join(path, dataset, "AW_distb.npy"))


    feat = normalize_features(feat)
    feat = torch.FloatTensor(feat)
    structure = torch.FloatTensor(structure)
    labels_tensor = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return g, nclass, feat, structure, structure, labels, labels_tensor, idx_train, idx_val, idx_test


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def evaluate(logits, labels):
    pred = np.argmax(logits, axis=1)
    macro_f1 = f1_score(labels, pred, average='macro')
    micro_f1 = f1_score(labels, pred, average='micro')
    return [macro_f1, micro_f1]


def sample_graph(g, img_edge_cnt, device, labels, is_compare):
    if img_edge_cnt == 0:
        g_new = g
    else:
        sample_src = np.random.randint(g.num_nodes(), size=img_edge_cnt)
        sample_dst = np.random.randint(g.num_nodes(), size=img_edge_cnt)
        print(sample_src)
        g_new = dgl.add_edges(g, sample_src, sample_dst)
        g_new = dgl.add_edges(g_new, sample_dst, sample_src)

    g_new = g_new.add_self_loop()

    e = g_new.edges()

    real_weight = torch.ones(g_new.num_edges(), dtype=int)
    if is_compare:
        cnt = 0
        for i in range(g_new.num_edges()):
            src = e[0][i]
            dst = e[1][i]
            if src == dst:
                real_weight[i] = 0
            if labels[src] != labels[dst]:
                real_weight[i] = -1
                cnt += 1
        print(cnt)
    
    g_new = g_new.to(device)
    return g_new, real_weight




