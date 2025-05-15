import re
import os
import torch
import subprocess
import numpy as np
import networkx as nx
from tqdm import tqdm
from os import makedirs, getcwd
from torch_geometric.data import Data
import torch_geometric.utils as tutils
from torch_geometric.utils.convert import from_networkx

import sys
sys.path.append('.')
from get_datasets import datasets

from experiments import bk, experiments

n_folds = 3

def load_file(filename):
    with open(filename, 'r') as file:
        return file.read()

def build_graph(literals, target):
    print("Creating graph...")
    
    dataset = []
    labels = []
    g_nodes, g_edges = [], []
    
    G = nx.Graph()
    
    n_negatives, n_positives = len(literals["neg"][:3]) + 1, len(literals["pos"][:3]) + 1
    all_literals = literals["neg"][:3] + [target + '(a,b)'] + literals["pos"][:3] + [target + '(a,b)']

    print('neg', literals['neg'][:3])
    print('pos', literals["pos"][:3])

    print(all_literals)
    relations, constants, numbers, nodes, edges = create_subhgraphs(all_literals)
    relations, constants, numbers, edges = list(set(relations)), list(set(constants)), list(set(numbers)), list(set(edges))
    distinct_nodes = sorted(relations) + sorted(constants) + numbers
    print('nodes', distinct_nodes)

    mapping_he = dict(zip(list(set(edges)), range(len(list(set(edges))))))
    mapping_v = dict(zip(list(set(distinct_nodes)), range(len(list(set(distinct_nodes))))))
    
    for i, (src, dst) in tqdm(enumerate(edges), desc='edges'):
        g_edges.append((mapping_v[src], mapping_v[dst]))
    
    for i, node in tqdm(enumerate(nodes), desc='mapping'):
        aux = np.zeros((len(distinct_nodes)))

        print('aaaaaaaaaaaa', node, target, i+1, n_negatives)
        if (node == target) and (i+1 <= n_negatives):
            aux[np.where(np.array(distinct_nodes) == node)] = -1 
        else:
            aux[np.where(np.array(distinct_nodes) == node)] = 1 
        print(node, aux)

        g_nodes.append((mapping_v[node], {'feature': aux}))
        labels.append(1 if node == target else 0)

    G.add_nodes_from(g_nodes)
    G.add_edges_from(g_edges)
        
    print('edges',edges)
    print('nodes',nodes)
    print('labels',labels)

    print(mapping_v)
    print(g_edges)
    dataset.append(from_networkx(G))
    print(dataset[0])

    return dataset, labels

def create_subhgraphs(literals):
    relations = []
    constants = []
    numbers = []
    edges = []
    nodes = []
    for dt in tqdm(literals, desc="feats"):

        relation = re.findall(r'([a-z\d_]+)\(', dt)[0]
        data = re.findall(r'\((.*?)\)', dt)

        nodes.append(relation)
        relations.append(relation)
        for nd in data[0].split(","):

            if len(nd.strip()) == 0: 
                continue

            edges.append((relation, nd))
            if nd.isdigit():
                numbers.append(nd)
            else:
                constants.append(nd)
            nodes.append(nd)          
    return relations, constants, numbers, nodes, edges

for experiment in experiments[:1]:

    datasets = datasets()

    results = {}
    source = experiment['source']
    predicate = experiment['predicate']

    print('Running experiment for %s' % source)

    PATH = f'''{os.getcwd() + '/test'}/{source}_{predicate}'''
    if not os.path.isdir(PATH):
        os.makedirs(PATH)

    filename = PATH + '/' + source + '_' + predicate

    # Load total target dataset
    tar_total_data = datasets.load(source, bk[source], seed=441773)

    if source in ['nell_sports', 'nell_finances', 'yago2s']:
        n_folds = n_folds
    else:
        n_folds = len(tar_total_data[0])

    for i in range(n_folds):
        print('\n Starting fold {} of {} folds \n'.format(i+1, n_folds))

        if source not in ['nell_sports', 'nell_finances', 'yago2s']:
            [tar_train_pos, tar_test_pos] = datasets.get_kfold_small(i, tar_total_data[0])
        else:
            t_total_data = datasets.load(source, bk[source], target=predicate, balanced=False, seed=441773)
            tar_train_pos = datasets.split_into_folds(t_total_data[1][0], n_folds=n_folds, seed=441773)[i] + t_total_data[0][0]

         # Load new predicate target dataset
        tar_data = datasets.load(source, bk[source], target=predicate, balanced=False, seed=441773)

        # Group and shuffle
        if source not in ['nell_sports', 'nell_finances', 'yago2s']:
            [tar_train_facts, tar_test_facts] =  datasets.get_kfold_small(i, tar_data[0])
            [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, tar_data[1])
            [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, tar_data[2])
        else:
            [tar_train_facts, tar_test_facts] =  [tar_data[0][0], tar_data[0][0]]
            to_folds_pos = datasets.split_into_folds(tar_data[1][0], n_folds=n_folds, seed=441773)
            to_folds_neg = datasets.split_into_folds(tar_data[2][0], n_folds=n_folds, seed=441773)
            [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, to_folds_pos)
            [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, to_folds_neg)

        graphs, label = build_graph({"pos": tar_train_facts + tar_train_pos, "neg": tar_train_facts + tar_train_neg}, predicate)
        print(graphs[0].number_of_nodes())
        print(graphs[0].number_of_edges())