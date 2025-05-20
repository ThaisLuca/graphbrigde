import os
import numpy as np
import random
import torch
from copy import deepcopy
from random import shuffle
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
import pickle as pk
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch import nn, optim

from os import listdir
from os.path import isfile, join

import torch.nn.functional as F

sizes = {}

for dataname in ["imdb", "uwcse", "yeast", "twitter", "nell_sports", "nell_finances", "cora"]:
    print("----- ", dataname.upper())
    mypath = 'dataset/{}/'.format(dataname)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    graph_list = []
    sizes[dataname] = []
    for i,file in enumerate(onlyfiles):
        print(file)
        dataset = torch.load(mypath + file, weights_only=False)
        data = Data(**dataset)

        #print('xxxx',dataset.data)
        #data = dataset.data
        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        sizes[dataname].append(data.x.shape[0])
        print(data.x.shape)
        print(edge_index.shape)
        input_dim = data.x.shape[1]
        hid_dim = input_dim
        graph_list.append(data)
        print('\n\n')

print(sizes)